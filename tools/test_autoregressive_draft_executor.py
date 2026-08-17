from __future__ import annotations

import importlib
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
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.autoregressive_draft_executor import (
    AutoregressiveDraftGroupExecution,
    AutoregressiveDraftProposalExecutor,
)
from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
    assert_tensor_free,
)

AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE = importlib.import_module(
    "tinyvllm.engine.autoregressive_draft_executor"
)


class _PhysicalStore:

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.next_slot = 0
        self.released = []

    def reserve_slots(self, count):
        slots = tuple(
            range(self.next_slot, self.next_slot + count)
        )
        self.next_slot += count
        return slots

    def release_slots(self, slot_ids):
        self.released.append(slot_ids)


class _RecordingAllocator(DirectProposalKVAllocator):

    def __init__(self, physical_store):
        super().__init__(physical_store)
        self.lease_calls = []

    def ensure_writable(self, identities):
        lease = super().ensure_writable(identities)
        self.lease_calls.append((
            "ensure_writable",
            identities,
            lease.physical_slot_ids,
        ))
        return lease

    def ensure_readable(self, identities):
        lease = super().ensure_readable(identities)
        self.lease_calls.append((
            "ensure_readable",
            identities,
            lease.physical_slot_ids,
        ))
        return lease

    def record_write_complete(self, lease):
        self.lease_calls.append((
            "record_write_complete",
            lease.identities,
            lease.physical_slot_ids,
        ))
        return super().record_write_complete(lease)

    def record_read_complete(self, lease):
        self.lease_calls.append((
            "record_read_complete",
            lease.identities,
            lease.physical_slot_ids,
        ))
        return super().record_read_complete(lease)


class _ResidencyStorage:

    def __init__(self, *, logical_capacity, gpu_capacity):
        self.logical_capacity = logical_capacity
        self.gpu_capacity = gpu_capacity
        self.block_size = 1
        self.dtype = torch.float32
        self.gpu_key_cache = torch.zeros(gpu_capacity, 2)
        self.gpu_value_cache = torch.zeros(gpu_capacity, 2)
        self.cpu_key_cache = torch.zeros(logical_capacity, 2)
        self.cpu_value_cache = torch.zeros(logical_capacity, 2)

    def entry_nbytes(self):
        return 16

    def copy_gpu_to_cpu(self, rows):
        for logical_entry_id, slot_id in rows:
            self.cpu_key_cache[logical_entry_id].copy_(
                self.gpu_key_cache[slot_id]
            )
            self.cpu_value_cache[logical_entry_id].copy_(
                self.gpu_value_cache[slot_id]
            )

    def copy_cpu_to_gpu(self, rows):
        for logical_entry_id, slot_id in rows:
            self.gpu_key_cache[slot_id].copy_(
                self.cpu_key_cache[logical_entry_id]
            )
            self.gpu_value_cache[slot_id].copy_(
                self.cpu_value_cache[logical_entry_id]
            )


class _RecordingResidencyAllocator(ProposalKVResidencyManager):

    def __init__(self, *, logical_capacity, gpu_capacity):
        super().__init__(
            storage=_ResidencyStorage(
                logical_capacity=logical_capacity,
                gpu_capacity=gpu_capacity,
            ),
            copy_backend=SynchronousProposalKVCopyBackend(),
        )
        self.writable_counts = []
        self.readable_counts = []

    def ensure_writable(self, identities):
        self.writable_counts.append(len(identities))
        return super().ensure_writable(identities)

    def ensure_blockwise_writable(self, identities):
        self.writable_counts.append(len(identities))
        return super().ensure_blockwise_writable(identities)

    def ensure_readable(self, identities):
        self.readable_counts.append(len(identities))
        return super().ensure_readable(identities)


class _Backend:
    device = torch.device("cpu")
    backend_identity = "fake-autoregressive"
    model_fingerprint = "model-sha256"
    tokenizer_fingerprint = "tokenizer-sha256"

    def prefill_batch(self, rows):
        raise AssertionError(
            "prefill is not used by observation tests"
        )

    def decode_step_batch(self, rows):
        raise AssertionError(
            "decode is not used by observation tests"
        )


def _executor():
    allocator = _RecordingAllocator(_PhysicalStore())
    return AutoregressiveDraftProposalExecutor(
        backend=_Backend(),
        proposal_kv_cache=ProposalKVCache(allocator),
        max_proposal_tokens=4,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
    )


def _row(
    token_ids,
    positions,
    *,
    final,
    epoch=7,
    target_hidden=None,
):
    if target_hidden is None:
        target_hidden = torch.full(
            (len(token_ids), 3),
            91.0,
        )
    return TargetPrefillObservation(
        sequence_id=11,
        sequence_epoch=epoch,
        token_ids=tuple(token_ids),
        positions=torch.tensor(positions),
        target_hidden=target_hidden,
        is_final_chunk=final,
    )


def test_capabilities_are_source_neutral_and_lifecycle_enabled():
    capabilities = _executor().capabilities

    assert capabilities.source_type == (
        "independent_draft_model"
    )
    assert capabilities.supports_batch is True
    assert capabilities.requires_target_hidden is False
    assert capabilities.requires_target_logits is False
    assert capabilities.max_proposal_tokens == 4
    assert capabilities.execution_domain == "model_runner"
    assert capabilities.requires_proposal_lifecycle is True
    assert capabilities.requires_full_token_history is False


def test_chunked_prefill_persists_only_tokens_positions_and_epoch():
    executor = _executor()

    executor.observe_target_prefill((
        _row((3, 4), (0, 1), final=False),
    ))
    executor.observe_target_prefill((
        _row((5,), (2,), final=True),
    ))

    pending = executor.pending_prompt(11)
    assert pending.sequence_epoch == 7
    assert pending.token_ids == (3, 4, 5)
    assert pending.positions == (0, 1, 2)
    assert pending.is_final is True
    assert not hasattr(pending, "target_hidden")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    (
        (
            _row((3,), (1,), final=True),
            None,
            "start at position zero",
        ),
        (
            _row((3,), (0,), final=False),
            _row((4,), (2,), final=True),
            "contiguous",
        ),
        (
            _row((3,), (0,), final=False),
            _row((4,), (1,), final=True, epoch=8),
            "epoch",
        ),
    ),
)
def test_invalid_prefill_chunks_fail_closed(
    first,
    second,
    message,
):
    executor = _executor()

    if second is None:
        with pytest.raises(
            (ValueError, RuntimeError),
            match=message,
        ):
            executor.observe_target_prefill((first,))
        return
    executor.observe_target_prefill((first,))
    with pytest.raises(
        (ValueError, RuntimeError),
        match=message,
    ):
        executor.observe_target_prefill((second,))


def test_duplicate_sequence_ids_fail_before_state_mutation():
    executor = _executor()
    row = _row((3,), (0,), final=True)

    with pytest.raises(ValueError, match="unique"):
        executor.observe_target_prefill((row, row))

    assert executor.pending_prompt(11) is None


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (
            TargetPrefillObservation(
                sequence_id=11,
                sequence_epoch=7,
                token_ids=(3, True),
                positions=torch.tensor((0, 1)),
                target_hidden=torch.zeros(2, 3),
                is_final_chunk=True,
            ),
            "token IDs",
        ),
        (
            TargetPrefillObservation(
                sequence_id=11,
                sequence_epoch=7,
                token_ids=(3,),
                positions=torch.tensor(((0,),)),
                target_hidden=torch.zeros(1, 3),
                is_final_chunk=True,
            ),
            "rank one",
        ),
        (
            TargetPrefillObservation(
                sequence_id=11,
                sequence_epoch=7,
                token_ids=(3,),
                positions=torch.tensor((0.0,)),
                target_hidden=torch.zeros(1, 3),
                is_final_chunk=True,
            ),
            "integer dtype",
        ),
        (
            _row(
                (3, 4),
                (0, 1),
                final=True,
                target_hidden=torch.zeros(1, 3),
            ),
            "row count",
        ),
    ),
)
def test_malformed_prefill_rows_fail_closed(row, message):
    executor = _executor()

    with pytest.raises(ValueError, match=message):
        executor.observe_target_prefill((row,))

    assert executor.pending_prompt(11) is None


def test_second_final_chunk_and_append_after_final_fail_closed():
    executor = _executor()
    executor.observe_target_prefill((
        _row((3,), (0,), final=True),
    ))

    with pytest.raises(RuntimeError, match="already final"):
        executor.observe_target_prefill((
            _row((4,), (1,), final=True),
        ))

    pending = executor.pending_prompt(11)
    assert pending.token_ids == (3,)
    assert pending.positions == (0,)


class _RecordingBackend:
    device = torch.device("cpu")
    backend_identity = "recording-backend"
    model_fingerprint = "model-sha256"
    tokenizer_fingerprint = "tokenizer-sha256"

    def __init__(
        self,
        *,
        fail_prefill_once=False,
        malformed_decode=None,
        tensor_parallel_rank=0,
        worker_returns_logits=False,
    ):
        self.prefill_calls = []
        self.decode_calls = []
        self.fail_prefill_once = fail_prefill_once
        self.malformed_decode = malformed_decode
        self.tensor_parallel_rank = tensor_parallel_rank
        self.worker_returns_logits = worker_returns_logits
        self.local_decode_forward_count = 0

    def prefill_batch(self, rows):
        self.prefill_calls.append(rows)
        if self.fail_prefill_once:
            self.fail_prefill_once = False
            raise RuntimeError("injected prefill failure")

    def decode_step_batch(self, rows):
        self.decode_calls.append(rows)
        self.local_decode_forward_count += 1
        if self.malformed_decode == "count":
            return ()
        if self.malformed_decode == "nonfinite":
            return tuple(
                torch.full((8,), float("nan"))
                for _ in rows
            )
        if (
            self.tensor_parallel_rank != 0
            and not self.worker_returns_logits
        ):
            return None
        outputs = []
        for row in rows:
            logits = torch.full((8,), -100.0)
            logits[(row.input_token_id + 1) % 8] = 10.0
            outputs.append(logits)
        return tuple(outputs)


class _RecordingGraphRunner:

    def __init__(self):
        self.calls = []
        self.eager_result_types = []
        self.convergence = None

    def bind_convergence(self, convergence):
        self.convergence = convergence

    def run(self, *, exact_q, rows, eager):
        self.calls.append((exact_q, rows))
        result = eager(exact_q, rows)
        self.eager_result_types.append(type(result))
        return result

    def summary(self):
        return {
            "captures": 0,
            "replays": len(self.calls),
            "ready_entries": (),
        }


class _CudaGraphRecordingRunner(_RecordingGraphRunner):

    def run(self, *, exact_q, rows, eager):
        self.calls.append((exact_q, rows))
        result = eager(exact_q, rows)
        self.eager_result_types.append(type(result))
        return AutoregressiveDraftGroupExecution(
            transactions=result.transactions,
            token_rows=result.token_rows,
            execution_mode="cuda_graph",
        )


class _RecordingCoordinator:

    def __init__(self):
        self.stages = []

    def assert_logical_authority(self, *, stage, rows):
        self.stages.append((stage, rows))
        return f"{stage}-digest"

    def converge_stage(self, *, stage, rows, local_error):
        self.stages.append((stage, rows))
        if local_error is not None:
            raise RuntimeError(f"{stage} failed") from local_error
        return f"{stage}-digest"


class _FailingCoordinator(_RecordingCoordinator):

    def __init__(self, failing_stage):
        super().__init__()
        self.failing_stage = failing_stage

    def assert_logical_authority(self, *, stage, rows):
        super().assert_logical_authority(
            stage=stage,
            rows=rows,
        )
        if stage == self.failing_stage:
            raise RuntimeError(f"{stage} mismatch")
        return f"{stage}-digest"

    def converge_stage(self, *, stage, rows, local_error):
        super().converge_stage(
            stage=stage,
            rows=rows,
            local_error=local_error,
        )
        if stage == self.failing_stage:
            raise RuntimeError(f"{stage} peer failure")
        return f"{stage}-digest"


def _sequence_row(
    sequence_id,
    token_ids,
    *,
    epoch=0,
):
    return TargetPrefillObservation(
        sequence_id=sequence_id,
        sequence_epoch=epoch,
        token_ids=tuple(token_ids),
        positions=torch.arange(
            len(token_ids),
            dtype=torch.int64,
        ),
        target_hidden=torch.zeros(len(token_ids), 3),
        is_final_chunk=True,
    )


def _ready_executor(
    prompts,
    *,
    backend=None,
    graph_runner=None,
    rank=0,
    world_size=1,
    coordinator=None,
    clock=None,
):
    if backend is None:
        backend = _RecordingBackend(
            tensor_parallel_rank=rank,
        )
    cache = ProposalKVCache(
        _RecordingAllocator(_PhysicalStore())
    )
    executor = AutoregressiveDraftProposalExecutor(
        backend=backend,
        proposal_kv_cache=cache,
        max_proposal_tokens=4,
        tensor_parallel_rank=rank,
        tensor_parallel_size=world_size,
        tensor_parallel_coordinator=coordinator,
        graph_runner=graph_runner,
        clock=clock,
    )
    executor.observe_target_prefill(tuple(
        _sequence_row(sequence_id, token_ids)
        for sequence_id, token_ids in prompts.items()
    ))
    return executor, backend, cache


def _ready_residency_executor(
    prompts,
    *,
    logical_capacity=32,
    gpu_capacity=8,
):
    backend = _RecordingBackend()
    allocator = _RecordingResidencyAllocator(
        logical_capacity=logical_capacity,
        gpu_capacity=gpu_capacity,
    )
    cache = ProposalKVCache(allocator)
    executor = AutoregressiveDraftProposalExecutor(
        backend=backend,
        proposal_kv_cache=cache,
        max_proposal_tokens=4,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
    )
    executor.observe_target_prefill(tuple(
        _sequence_row(sequence_id, token_ids)
        for sequence_id, token_ids in prompts.items()
    ))
    return executor, backend, cache, allocator


def _proposal_input(
    sequence_id,
    *,
    first_target_token,
    exact_q,
    context_token_count,
):
    return ModelRunnerProposalInput(
        sequence_id=sequence_id,
        token_ids=(),
        remaining_output_tokens=exact_q,
        max_proposal_tokens=4,
        first_target_token=first_target_token,
        context_token_count=context_token_count,
    )


def test_q4_batch_four_dispatches_through_graph_runner_then_registers():
    graph_runner = _RecordingGraphRunner()
    executor, backend, cache = _ready_executor(
        {
            1: (2, 3),
            2: (3, 4),
            3: (4, 5),
            4: (5, 6),
        },
        graph_runner=graph_runner,
    )

    proposals = executor.propose_batch(tuple(
        _proposal_input(
            sequence_id,
            first_target_token=sequence_id + 5,
            exact_q=4,
            context_token_count=2,
        )
        for sequence_id in range(1, 5)
    ))

    assert len(graph_runner.calls) == 1
    exact_q, rows = graph_runner.calls[0]
    assert exact_q == 4
    assert tuple(
        row[1].sequence_id for row in rows
    ) == (1, 2, 3, 4)
    assert graph_runner.eager_result_types == [
        AutoregressiveDraftGroupExecution
    ]
    assert len(backend.decode_calls) == 3
    assert all(
        proposal.proposal_transaction_id is not None
        for proposal in proposals
    )
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 4

    ticket = executor.prepare_finalize_batch(tuple(
        _finalize_row(
            proposal,
            accepted=4 if index < 2 else 0,
        )
        for index, proposal in enumerate(proposals)
    ))
    executor.commit_finalize_batch(ticket)
    assert tuple(
        cache.committed_length(sequence_id)
        for sequence_id in range(1, 5)
    ) == (5, 5, 2, 2)
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_graph_path_does_not_change_materialized_authority_digest_rows():
    prompts = {
        1: (2, 3),
        2: (3, 4),
        3: (4, 5),
        4: (5, 6),
    }
    inputs = tuple(
        _proposal_input(
            sequence_id,
            first_target_token=sequence_id + 5,
            exact_q=4,
            context_token_count=2,
        )
        for sequence_id in range(1, 5)
    )
    eager_coordinator = _RecordingCoordinator()
    graph_coordinator = _RecordingCoordinator()
    eager_executor, _, _ = _ready_executor(
        prompts,
        coordinator=eager_coordinator,
    )
    graph_executor, _, _ = _ready_executor(
        prompts,
        coordinator=graph_coordinator,
        graph_runner=_CudaGraphRecordingRunner(),
    )

    eager_executor.propose_batch(inputs)
    graph_executor.propose_batch(inputs)

    eager_rows = next(
        rows
        for stage, rows in eager_coordinator.stages
        if stage == "proposal_materialized"
    )
    graph_rows = next(
        rows
        for stage, rows in graph_coordinator.stages
        if stage == "proposal_materialized"
    )
    assert graph_rows == eager_rows


def test_q1_bypasses_graph_runner():
    graph_runner = _RecordingGraphRunner()
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        graph_runner=graph_runner,
    )

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=1,
            context_token_count=2,
        ),
    ))[0]

    assert proposal.token_ids == (4,)
    assert graph_runner.calls == []


def test_authority_snapshot_includes_tensor_free_graph_summary():
    graph_runner = _RecordingGraphRunner()
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        graph_runner=graph_runner,
    )

    snapshot = executor.authority_snapshot()

    assert snapshot["cuda_graph"] == {
        "captures": 0,
        "replays": 0,
        "ready_entries": (),
    }
    assert_tensor_free(
        snapshot,
        name="executor graph authority snapshot",
    )


def test_executor_binds_graph_convergence_to_tp_coordinator():
    graph_runner = _RecordingGraphRunner()
    coordinator = _RecordingCoordinator()
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        graph_runner=graph_runner,
        coordinator=coordinator,
    )

    assert graph_runner.convergence is not None
    digest = graph_runner.convergence(
        stage="graph_pre_replay",
        rows={"exact_q": 4},
        local_error=None,
    )

    assert digest == "graph_pre_replay-digest"
    assert coordinator.stages[-1] == (
        "graph_pre_replay",
        {"exact_q": 4},
    )
    assert executor.authority_snapshot()[
        "logical_authority_rows"
    ][-1]["stage"] == "graph_pre_replay"


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_executor_accepts_every_valid_rank(rank):
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        rank=rank,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )

    assert executor.tensor_parallel_rank == rank
    assert executor.tensor_parallel_size == 4


@pytest.mark.parametrize("world_size", (2, 3, 5, 8))
def test_executor_rejects_every_other_topology(world_size):
    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        _ready_executor(
            {1: (2, 3)},
            rank=0,
            world_size=world_size,
            coordinator=_RecordingCoordinator(),
        )


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_root_and_workers_select_identical_tokens(
    rank,
    monkeypatch,
):
    coordinator = _RecordingCoordinator()
    executor, backend, _ = _ready_executor(
        {1: (2, 3)},
        rank=rank,
        world_size=4,
        coordinator=coordinator,
    )
    calls = []

    def select(logits, **kwargs):
        calls.append((logits, kwargs))
        return torch.tensor([7], dtype=torch.int64)

    monkeypatch.setattr(
        AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE,
        "select_tensor_parallel_greedy_tokens",
        select,
    )

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]

    assert proposal.token_ids == (4, 7)
    assert len(calls) == 1
    if rank == 0:
        assert calls[0][0].shape == (1, 8)
    else:
        assert calls[0][0] is None
    assert calls[0][1]["rank"] == rank
    assert calls[0][1]["world_size"] == 4
    assert backend.local_decode_forward_count == 1


def test_tp4_non_root_local_vocabulary_logits_fail_closed(
    monkeypatch,
):
    backend = _RecordingBackend(
        tensor_parallel_rank=2,
        worker_returns_logits=True,
    )
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        backend=backend,
        rank=2,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )
    monkeypatch.setattr(
        AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE,
        "select_tensor_parallel_greedy_tokens",
        lambda logits, **kwargs: torch.tensor(
            [7],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="proposal_decode_step_0",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=2,
                context_token_count=2,
            ),
        ))

    assert cache.committed_length(1) == 2
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_tp4_broadcast_failure_aborts_new_transaction(
    monkeypatch,
):
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        rank=0,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )

    def fail_broadcast(*args, **kwargs):
        raise RuntimeError("injected broadcast failure")

    monkeypatch.setattr(
        AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE,
        "select_tensor_parallel_greedy_tokens",
        fail_broadcast,
    )

    with pytest.raises(
        RuntimeError,
        match="proposal_decode_step_0",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=3,
                context_token_count=2,
            ),
        ))

    assert cache.committed_length(1) == 2
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_proposal_preflight_mismatch_allocates_no_slots():
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        rank=0,
        world_size=4,
        coordinator=_FailingCoordinator(
            "proposal_preflight"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="proposal_preflight",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=4,
                context_token_count=2,
            ),
        ))

    assert cache.authority_snapshot()["owned_entry_count"] == 0
    assert cache.sequence_state(1) is None


def test_proposal_materialized_mismatch_aborts_before_registration(
    monkeypatch,
):
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        rank=0,
        world_size=4,
        coordinator=_FailingCoordinator(
            "proposal_materialized"
        ),
    )
    monkeypatch.setattr(
        AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE,
        "select_tensor_parallel_greedy_tokens",
        lambda logits, **kwargs: torch.tensor(
            [7],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="proposal_materialized",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=2,
                context_token_count=2,
            ),
        ))

    assert cache.committed_length(1) == 2
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0
    assert cache.authority_snapshot()["owned_entry_count"] == 2


@pytest.mark.parametrize("exact_q", (1, 2, 3, 4))
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_exact_q_preserves_q_minus_one_staged_entries(
    exact_q,
    rank,
    monkeypatch,
):
    executor, backend, cache = _ready_executor(
        {1: (2, 3)},
        rank=rank,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )
    monkeypatch.setattr(
        AUTOREGRESSIVE_DRAFT_EXECUTOR_MODULE,
        "select_tensor_parallel_greedy_tokens",
        lambda logits, **kwargs: torch.tensor(
            [7],
            dtype=torch.int64,
        ),
    )

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=exact_q,
            context_token_count=2,
        ),
    ))[0]

    assert proposal.metadata["staged_entry_count"] == exact_q - 1
    transaction = cache.transaction(
        proposal.proposal_transaction_id
    )
    assert len(transaction.staged_entry_identities) == exact_q - 1
    assert backend.local_decode_forward_count == max(exact_q - 1, 0)


def test_first_proposal_bootstraps_prompt_once_and_batches_decode():
    executor, backend, cache = _ready_executor({
        1: (2, 3),
        2: (4, 5, 6),
    })

    proposals = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=6,
            exact_q=4,
            context_token_count=2,
        ),
        _proposal_input(
            2,
            first_target_token=2,
            exact_q=4,
            context_token_count=3,
        ),
    ))

    assert len(backend.prefill_calls) == 1
    assert tuple(
        row.token_ids
        for row in backend.prefill_calls[0]
    ) == ((2, 3), (4, 5, 6))
    assert [len(call) for call in backend.decode_calls] == [
        2,
        2,
        2,
    ]
    assert proposals[0].token_ids == (6, 7, 0, 1)
    assert proposals[1].token_ids == (2, 3, 4, 5)
    assert cache.committed_length(1) == 2
    assert cache.committed_length(2) == 3

    executor.proposal_kv_lifecycle.rollback_finalize_batch(
        executor.proposal_kv_lifecycle.prepare_finalize_batch((
            _finalize_row(proposals[0], accepted=0),
            _finalize_row(proposals[1], accepted=0),
        ))
    )
    executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=3,
            exact_q=1,
            context_token_count=2,
        ),
    ))
    assert len(backend.prefill_calls) == 1


def test_residency_bootstrap_streams_prompt_one_token_per_decode_round():
    executor, backend, cache, allocator = (
        _ready_residency_executor(
            {1: (2, 3, 4, 5, 6)},
            logical_capacity=32,
            gpu_capacity=2,
        )
    )

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=7,
            exact_q=1,
            context_token_count=5,
        ),
    ))[0]

    assert proposal.token_ids == (7,)
    assert backend.prefill_calls == []
    assert len(backend.decode_calls) == 5
    assert all(len(call) == 1 for call in backend.decode_calls)
    assert [
        len(call[0].visible_logical_entry_ids)
        for call in backend.decode_calls
    ] == [1, 2, 3, 4, 5]
    assert all(
        call[0].blockwise_offload
        for call in backend.decode_calls
    )
    assert allocator.writable_counts == [1, 1, 1, 1, 1]
    assert allocator.readable_counts == []
    assert cache.committed_length(1) == 5


def test_residency_batch_four_decode_does_not_pin_full_history():
    executor, backend, cache, allocator = (
        _ready_residency_executor(
            {
                1: (2, 3, 4),
                2: (3, 4, 5),
                3: (4, 5, 6),
                4: (5, 6, 7),
            },
            logical_capacity=32,
            gpu_capacity=8,
        )
    )
    bootstrap_proposals = executor.propose_batch(tuple(
        _proposal_input(
            sequence_id,
            first_target_token=sequence_id + 8,
            exact_q=1,
            context_token_count=3,
        )
        for sequence_id in range(1, 5)
    ))
    bootstrap_ticket = executor.prepare_finalize_batch(tuple(
        _finalize_row(proposal, accepted=1)
        for proposal in bootstrap_proposals
    ))
    executor.commit_finalize_batch(bootstrap_ticket)
    backend.decode_calls.clear()
    allocator.writable_counts.clear()
    allocator.readable_counts.clear()

    proposals = executor.propose_batch(tuple(
        _proposal_input(
            sequence_id,
            first_target_token=sequence_id + 12,
            exact_q=4,
            context_token_count=3,
        )
        for sequence_id in range(1, 5)
    ))

    assert len(backend.decode_calls) == 3
    assert all(
        len(decode_rows) == 4
        for decode_rows in backend.decode_calls
    )
    assert all(
        row.blockwise_offload
        for decode_rows in backend.decode_calls
        for row in decode_rows
    )
    assert [
        len(row.visible_logical_entry_ids)
        for decode_rows in backend.decode_calls
        for row in decode_rows
    ] == [4] * 4 + [5] * 4 + [6] * 4
    assert all(
        len(row.visible_physical_slot_ids) == 1
        for decode_rows in backend.decode_calls
        for row in decode_rows
    )
    assert allocator.writable_counts == [1] * 12
    assert allocator.readable_counts == []

    finalize_ticket = executor.prepare_finalize_batch(tuple(
        _finalize_row(
            proposal,
            accepted=4 if index < 2 else 0,
        )
        for index, proposal in enumerate(proposals)
    ))
    executor.commit_finalize_batch(finalize_ticket)

    assert tuple(
        cache.committed_length(sequence_id)
        for sequence_id in range(1, 5)
    ) == (6, 6, 3, 3)
    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_executor_maps_logical_entries_through_forward_leases():
    executor, backend, cache = _ready_executor({
        1: (2, 3),
    })

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=6,
            exact_q=3,
            context_token_count=2,
        ),
    ))[0]

    assert proposal.token_ids == (6, 7, 0)
    prefill_row = backend.prefill_calls[0][0]
    assert prefill_row.physical_slot_ids == (0, 1)
    first_decode = backend.decode_calls[0][0]
    second_decode = backend.decode_calls[1][0]
    assert first_decode.writable_physical_slot_id == 2
    assert first_decode.visible_physical_slot_ids == (0, 1, 2)
    assert second_decode.writable_physical_slot_id == 3
    assert second_decode.visible_physical_slot_ids == (0, 1, 2, 3)
    assert tuple(
        call[0] for call in cache.entry_allocator.lease_calls
    ) == (
        "ensure_writable",
        "record_write_complete",
        "ensure_readable",
        "ensure_writable",
        "record_read_complete",
        "record_write_complete",
        "ensure_readable",
        "ensure_writable",
        "record_read_complete",
        "record_write_complete",
    )


def _finalize_row(proposal, *, accepted):
    return ProposalFinalizeRow(
        sequence_id=proposal.sequence_id,
        proposal_transaction_id=(
            proposal.proposal_transaction_id
        ),
        accepted_proposal_tokens=accepted,
    )


def test_mixed_q_preserves_order_and_q_zero_skips_bootstrap():
    executor, backend, cache = _ready_executor({
        1: (1,),
        2: (2,),
        3: (3,),
        4: (4,),
        5: (5,),
    })

    proposals = executor.propose_batch(tuple(
        _proposal_input(
            sequence_id,
            first_target_token=sequence_id,
            exact_q=exact_q,
            context_token_count=1,
        )
        for sequence_id, exact_q in zip(
            range(1, 6),
            range(5),
        )
    ))

    assert tuple(
        proposal.sequence_id for proposal in proposals
    ) == (1, 2, 3, 4, 5)
    assert tuple(
        len(proposal.token_ids) for proposal in proposals
    ) == (0, 1, 2, 3, 4)
    assert proposals[0].proposal_transaction_id is None
    assert cache.sequence_state(1) is None
    assert tuple(
        row.transaction.sequence_id
        for row in backend.prefill_calls[0]
    ) == (2, 3, 4, 5)
    for proposal in proposals[1:]:
        transaction = cache.transaction(
            proposal.proposal_transaction_id
        )
        assert len(transaction.staged_entry_identities) == (
            len(proposal.token_ids) - 1
        )
        assert proposal.token_ids[0] == proposal.sequence_id


@pytest.mark.parametrize(
    ("malformed_decode", "message"),
    (
        ("count", "logit row count"),
        ("nonfinite", "finite"),
    ),
)
def test_decode_contract_failure_aborts_new_proposal_transaction(
    malformed_decode,
    message,
):
    backend = _RecordingBackend(
        malformed_decode=malformed_decode,
    )
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        backend=backend,
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=3,
                context_token_count=2,
            ),
        ))

    assert cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0
    assert cache.authority_snapshot()["owned_entry_count"] == 2


def test_bootstrap_failure_releases_slots_and_is_retryable():
    backend = _RecordingBackend(fail_prefill_once=True)
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        backend=backend,
    )
    input_row = _proposal_input(
        1,
        first_target_token=4,
        exact_q=2,
        context_token_count=2,
    )

    with pytest.raises(RuntimeError, match="prefill failure"):
        executor.propose_batch((input_row,))

    assert cache.authority_snapshot()["owned_entry_count"] == 0
    assert executor.pending_prompt(1).token_ids == (2, 3)

    proposal = executor.propose_batch((input_row,))[0]
    assert proposal.token_ids == (4, 5)
    assert cache.committed_length(1) == 2


def test_bootstrap_preflight_mismatch_allocates_no_slots():
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator(
            "bootstrap_preflight"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="bootstrap_preflight",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=4,
                context_token_count=2,
            ),
        ))

    snapshot = cache.authority_snapshot()
    assert snapshot["active_transaction_count"] == 0
    assert snapshot["owned_entry_count"] == 0


def test_peer_bootstrap_prepare_failure_rolls_back_in_reverse_order():
    executor, _, cache = _ready_executor(
        {
            1: (2, 3),
            2: (4, 5, 6),
        },
        coordinator=_FailingCoordinator(
            "bootstrap_prepared"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="bootstrap_prepared",
    ):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=4,
                context_token_count=2,
            ),
            _proposal_input(
                2,
                first_target_token=6,
                exact_q=4,
                context_token_count=3,
            ),
        ))

    snapshot = cache.authority_snapshot()
    assert cache.committed_length(1) == 0
    assert cache.committed_length(2) == 0
    assert snapshot["active_transaction_count"] == 0
    assert snapshot["prepared_ticket_count"] == 0
    assert snapshot["owned_entry_count"] == 0
    assert cache.entry_allocator.physical_store.released == [
        (2, 3, 4),
        (0, 1),
    ]


def test_finalize_preflight_mismatch_calls_no_local_prepare(
    monkeypatch,
):
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator(
            "finalize_preflight"
        ),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]

    def unexpected_prepare(rows):
        raise AssertionError("local prepare must not run")

    monkeypatch.setattr(
        executor.proposal_kv_lifecycle,
        "prepare_finalize_batch",
        unexpected_prepare,
    )

    with pytest.raises(
        RuntimeError,
        match="finalize_preflight",
    ):
        executor.prepare_finalize_batch((
            _finalize_row(proposal, accepted=2),
        ))


@pytest.mark.parametrize(
    "failing_stage",
    ("finalize_prepare", "finalize_prepared"),
)
def test_peer_finalize_prepare_failure_cleans_local_state(
    failing_stage,
):
    executor, _, cache = _ready_executor(
        {
            1: (2, 3),
            2: (4, 5, 6),
        },
        coordinator=_FailingCoordinator(failing_stage),
    )
    proposals = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
        _proposal_input(
            2,
            first_target_token=6,
            exact_q=4,
            context_token_count=3,
        ),
    ))

    with pytest.raises(RuntimeError, match=failing_stage):
        executor.prepare_finalize_batch((
            _finalize_row(proposals[0], accepted=3),
            _finalize_row(proposals[1], accepted=2),
        ))

    snapshot = cache.authority_snapshot()
    lifecycle = (
        executor.proposal_kv_lifecycle.authority_snapshot()
    )
    assert cache.committed_length(1) == 2
    assert cache.committed_length(2) == 3
    assert snapshot["active_transaction_count"] == 0
    assert snapshot["prepared_ticket_count"] == 0
    assert lifecycle["active_transaction_count"] == 0
    assert lifecycle["prepared_ticket_count"] == 0


def test_local_finalize_commit_failure_is_poisoned(monkeypatch):
    executor, _, _ = _ready_executor({1: (2, 3)})
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=3),
    ))

    def fail_commit(_ticket_id):
        raise RuntimeError("local commit failure")

    monkeypatch.setattr(
        executor.proposal_kv_lifecycle,
        "commit_finalize_batch",
        fail_commit,
    )

    with pytest.raises(
        RuntimeError,
        match="poisoned after finalize commit",
    ):
        executor.commit_finalize_batch(ticket_id)


def test_finalize_committed_mismatch_is_poisoned():
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator(
            "finalize_committed"
        ),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=3),
    ))

    with pytest.raises(
        RuntimeError,
        match="poisoned after finalize commit",
    ):
        executor.commit_finalize_batch(ticket_id)

    assert cache.committed_length(1) == 4


def test_finalize_rollback_compares_logical_state_and_retains_prompt():
    coordinator = _RecordingCoordinator()
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=coordinator,
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=3),
    ))

    executor.rollback_finalize_batch(ticket_id)

    assert cache.committed_length(1) == 2
    stage_rows = dict(coordinator.stages)
    assert stage_rows["finalize_rolled_back"] == ({
        "batch_index": 0,
        "sequence_id": 1,
        "sequence_epoch": 0,
        "exact_q": 4,
        "proposal_token_ids": proposal.token_ids,
        "accepted_proposal_tokens": 3,
        "committed_proposal_entries": 2,
        "logical_state": "rolled_back",
    },)


def test_peer_finalize_rollback_failure_is_poisoned():
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator(
            "finalize_rolled_back"
        ),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=3),
    ))

    with pytest.raises(
        RuntimeError,
        match="poisoned after finalize rollback",
    ):
        executor.rollback_finalize_batch(ticket_id)


def test_release_preflight_mismatch_preserves_sequence_state():
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator(
            "release_preflight"
        ),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=0),
    ))
    executor.rollback_finalize_batch(ticket_id)

    with pytest.raises(
        RuntimeError,
        match="release_preflight",
    ):
        executor.release_sequence(1, sequence_epoch=0)

    assert executor.pending_prompt(1) is not None
    assert cache.committed_length(1) == 2


def test_successful_release_compares_zero_complete_state():
    coordinator = _RecordingCoordinator()
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=coordinator,
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=0),
    ))
    executor.rollback_finalize_batch(ticket_id)

    executor.release_sequence(1, sequence_epoch=0)

    stage_rows = dict(coordinator.stages)
    assert stage_rows["release_complete"] == ({
        "sequence_id": 1,
        "sequence_epoch": 0,
        "active_transaction_count": 0,
        "active_ticket_count": 0,
        "committed_logical_entries": 0,
        "live_local_slot_count": 0,
    },)
    assert cache.sequence_state(1) is None
    assert cache.authority_snapshot()["owned_entry_count"] == 0


def test_peer_release_failure_is_poisoned_and_not_reusable():
    executor, _, cache = _ready_executor(
        {1: (2, 3)},
        coordinator=_FailingCoordinator("release_local"),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=0),
    ))
    executor.rollback_finalize_batch(ticket_id)

    with pytest.raises(
        RuntimeError,
        match="poisoned after release",
    ):
        executor.release_sequence(1, sequence_epoch=0)

    assert executor.pending_prompt(1) is not None
    assert cache.sequence_state(1) is None


@pytest.mark.parametrize(
    "accepted",
    (0, 1, 2, 3, 4),
)
def test_partial_acceptance_commits_exact_accepted_minus_one_entries(
    accepted,
):
    executor, _, cache = _ready_executor({1: (2, 3)})
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]

    ticket = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=accepted),
    ))
    executor.commit_finalize_batch(ticket)

    assert cache.committed_length(1) == (
        2 + max(accepted - 1, 0)
    )


def test_rollback_releases_staged_suffix_and_retains_prompt():
    executor, _, cache = _ready_executor({1: (2, 3)})
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]

    ticket = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=3),
    ))
    executor.rollback_finalize_batch(ticket)

    assert cache.committed_length(1) == 2
    assert cache.authority_snapshot()["owned_entry_count"] == 2


def test_multiple_accepted_rounds_append_without_bootstrap_replay():
    executor, backend, cache = _ready_executor({1: (2, 3)})
    first = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=4,
            context_token_count=2,
        ),
    ))[0]
    first_ticket = executor.prepare_finalize_batch((
        _finalize_row(first, accepted=3),
    ))
    executor.commit_finalize_batch(first_ticket)

    second = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=5,
            exact_q=3,
            context_token_count=4,
        ),
    ))[0]
    second_ticket = executor.prepare_finalize_batch((
        _finalize_row(second, accepted=2),
    ))
    executor.commit_finalize_batch(second_ticket)

    assert len(backend.prefill_calls) == 1
    assert cache.committed_length(1) == 5


def test_release_rejects_active_transaction_then_clears_all_state():
    executor, _, cache = _ready_executor({1: (2, 3)})
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=3,
            context_token_count=2,
        ),
    ))[0]

    with pytest.raises(RuntimeError, match="active"):
        executor.release_sequence(1, sequence_epoch=0)

    ticket = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=0),
    ))
    executor.rollback_finalize_batch(ticket)
    executor.release_sequence(1, sequence_epoch=0)

    assert executor.pending_prompt(1) is None
    assert cache.sequence_state(1) is None
    assert cache.authority_snapshot()["owned_entry_count"] == 0


def test_release_rejects_stale_epoch():
    executor, _, _ = _ready_executor({1: (2, 3)})

    with pytest.raises(RuntimeError, match="epoch"):
        executor.release_sequence(1, sequence_epoch=1)


def test_authority_snapshot_is_tensor_free_and_complete():
    executor, _, _ = _ready_executor({1: (2, 3)})
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]

    snapshot = executor.authority_snapshot()

    assert_tensor_free(snapshot, name="autoregressive snapshot")
    assert snapshot["source_type"] == "independent_draft_model"
    assert snapshot["backend_identity"] == "recording-backend"
    assert snapshot["model_fingerprint"] == "model-sha256"
    assert snapshot["tokenizer_fingerprint"] == (
        "tokenizer-sha256"
    )
    assert snapshot["tensor_parallel_rank"] == 0
    assert snapshot["tensor_parallel_size"] == 1
    assert snapshot["bootstrap_rows"] == ({
        "sequence_id": 1,
        "sequence_epoch": 0,
        "prompt_token_count": 2,
        "bootstrap_commit_encoding": 3,
    },)
    assert snapshot["proposal_exact_q"] == ({
        "transaction_id": proposal.proposal_transaction_id,
        "exact_q": 2,
    },)
    assert set(snapshot["timing_ms"]) == {
        "prompt_bootstrap",
        "proposal_forward",
        "proposal_finalize",
    }
    assert snapshot["timing_ms"]["prompt_bootstrap"] > 0.0
    assert snapshot["timing_ms"]["proposal_forward"] > 0.0
    assert snapshot["timing_ms"]["proposal_finalize"] == 0.0
    assert snapshot["proposal_kv_lifecycle"][
        "active_transaction_count"
    ] == 1


def test_executor_accumulates_bootstrap_forward_and_finalize_timing():
    clock_value = -0.001

    def clock():
        nonlocal clock_value
        clock_value += 0.001
        return clock_value

    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        clock=clock,
    )

    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]
    ticket = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=1),
    ))
    executor.commit_finalize_batch(ticket)

    snapshot = executor.authority_snapshot()

    assert snapshot["proposal_forward_detail_ms"] == {
        "setup": pytest.approx(2.0),
        "backend_submit": pytest.approx(1.0),
        "selection_collective": pytest.approx(1.0),
        "decode_authority": pytest.approx(0.0),
        "token_readback": pytest.approx(1.0),
        "materialize_register": pytest.approx(1.0),
    }
    assert snapshot["timing_ms"] == {
        "prompt_bootstrap": pytest.approx(1.0),
        "proposal_forward": pytest.approx(13.0),
        "proposal_finalize": pytest.approx(2.0),
    }


def test_authority_snapshot_records_logical_stage_evidence():
    coordinator = _RecordingCoordinator()
    executor, _, _ = _ready_executor(
        {1: (2, 3)},
        coordinator=coordinator,
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=2,
            context_token_count=2,
        ),
    ))[0]
    ticket_id = executor.prepare_finalize_batch((
        _finalize_row(proposal, accepted=0),
    ))
    executor.rollback_finalize_batch(ticket_id)
    executor.release_sequence(1, sequence_epoch=0)

    snapshot = executor.authority_snapshot()

    assert_tensor_free(
        snapshot,
        name="autoregressive logical authority snapshot",
    )
    assert snapshot["rank"] == 0
    assert snapshot["world_size"] == 1
    assert isinstance(snapshot["logical_authority_rows"], tuple)
    assert snapshot["logical_authority_digest_count"] == len(
        snapshot["logical_authority_rows"]
    )
    assert snapshot["last_logical_authority_sha256"] == (
        "release_complete-digest"
    )
    assert tuple(
        row["stage"]
        for row in snapshot["logical_authority_rows"]
    )[-3:] == (
        "release_preflight",
        "release_local",
        "release_complete",
    )
    assert snapshot["logical_authority_rows"][-1] == {
        "stage": "release_complete",
        "rows": ({
            "sequence_id": 1,
            "sequence_epoch": 0,
            "active_transaction_count": 0,
            "active_ticket_count": 0,
            "committed_logical_entries": 0,
            "live_local_slot_count": 0,
        },),
    }
