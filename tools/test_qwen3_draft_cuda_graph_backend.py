from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from types import SimpleNamespace

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
)
from tinyvllm.engine.autoregressive_draft_graph import (
    AutoregressiveDraftGraphEntry,
    AutoregressiveDraftGraphIdentity,
    AutoregressiveDraftGraphPreReplayError,
)
from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.qwen3_draft_cuda_graph_backend import (
    Qwen3DraftCudaGraphBackend,
    Qwen3DraftCudaGraphPayload,
)
from tinyvllm.engine.qwen3_draft_graph_scratch import (
    Qwen3DraftGraphScratchOwner,
)
from tinyvllm.engine.qwen3_draft_proposal_kv import (
    Qwen3DraftPhysicalSlotStore,
)


class _FakeGraph:

    def __init__(self, replay_callback=None):
        self.replay_callback = replay_callback
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1
        if self.replay_callback is not None:
            self.replay_callback()


class _GraphContext:

    def __init__(self, owner, graph):
        self.owner = owner
        self.graph = graph

    def __enter__(self):
        self.owner.graph_entries += 1
        return self.graph

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeCuda:

    def __init__(self):
        self.graph_entries = 0
        self.synchronize_calls = 0

    def graph_pool_handle(self):
        return "graph-pool"

    def Stream(self):
        return "capture-stream"

    def CUDAGraph(self):
        return _FakeGraph()

    def graph(self, graph, *, pool, stream):
        assert pool == "graph-pool"
        assert stream == "capture-stream"
        return _GraphContext(self, graph)

    def synchronize(self):
        self.synchronize_calls += 1

    @staticmethod
    def memory_allocated():
        return 100

    @staticmethod
    def memory_reserved():
        return 200


class _FakeTorch:
    int64 = torch.int64
    int32 = torch.int32

    def __init__(self):
        self.cuda = _FakeCuda()

    @staticmethod
    def zeros(*args, **kwargs):
        return torch.zeros(*args, **kwargs)

    @staticmethod
    def tensor(*args, **kwargs):
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def argmax(*args, **kwargs):
        return torch.argmax(*args, **kwargs)


class _StaticDraftBackend:

    def __init__(self, *, rank=0):
        self.device = torch.device("cpu")
        self.tensor_parallel_rank = rank
        self.tensor_parallel_size = 4
        self.compute_dtype = torch.float32
        self.calls = []

    def decode_step_static(
        self,
        input_ids,
        positions,
        slot_mapping,
        context_lens,
        block_tables,
    ):
        self.calls.append({
            "input_ids": input_ids.clone(),
            "positions": positions.clone(),
            "slot_mapping": slot_mapping.clone(),
            "context_lens": context_lens.clone(),
            "block_tables": block_tables.clone(),
        })
        if self.tensor_parallel_rank != 0:
            return None
        selected = ((len(self.calls) - 1) % 3) + 5
        logits = torch.full(
            (input_ids.shape[0], 16),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, selected] = 10.0
        return logits


class _Broadcast:

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, tensor, *, src):
        self.calls.append((tensor, src))
        if self.fail:
            raise RuntimeError("injected broadcast failure")


class _TrackingProposalTokens:

    def __init__(self, values):
        self.tensor = torch.tensor(values, dtype=torch.int64)
        self.shape = tuple(self.tensor.shape)
        self.dtype = torch.int64
        self.device = torch.device("cpu")
        self.tolist_calls = 0

    def copy_(self, source):
        self.tensor.copy_(source)
        return self

    def zero_(self):
        self.tensor.zero_()
        return self

    def __getitem__(self, item):
        return self.tensor[item]

    def tolist(self):
        self.tolist_calls += 1
        return self.tensor.tolist()


def _shape_model():
    layers = []
    for _ in range(2):
        attention_backend = SimpleNamespace(
            k_cache=torch.Tensor(),
            v_cache=torch.Tensor(),
            kv_quant_bits=None,
        )
        attention = SimpleNamespace(
            num_heads=8,
            num_kv_heads=2,
            head_dim=4,
            attn=attention_backend,
        )
        layers.append(SimpleNamespace(self_attn=attention))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _caches():
    physical_store = Qwen3DraftPhysicalSlotStore(
        _shape_model(),
        capacity=128,
        dtype=torch.float32,
        device="cpu",
    )
    live_cache = ProposalKVCache(
        DirectProposalKVAllocator(physical_store)
    )
    scratch_cache = ProposalKVCache(
        DirectProposalKVAllocator(physical_store)
    )
    return live_cache, scratch_cache


def _commit_prompt(cache, sequence_id, token_count):
    transaction = cache.begin(sequence_id, 0, token_count)
    write_lease = cache.entry_allocator.ensure_writable(
        transaction.staged_entry_identities
    )
    cache.entry_allocator.record_write_complete(write_lease)
    cache.mark_materialized(transaction, token_count)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=token_count + 1,
    )
    cache.commit_finalize(ticket.ticket_id)


def _indexed_rows(live_cache):
    rows = []
    for index, (sequence_id, context_count, first_token) in enumerate(
        ((7, 2, 31), (9, 1, 32), (11, 3, 33), (13, 1, 34))
    ):
        _commit_prompt(live_cache, sequence_id, context_count)
        rows.append((
            index,
            SimpleNamespace(
                sequence_id=sequence_id,
                first_target_token=first_token,
            ),
            context_count,
        ))
    return tuple(rows)


def _identity():
    return AutoregressiveDraftGraphIdentity(
        exact_q=4,
        exact_batch_size=4,
        tensor_parallel_size=4,
        tensor_parallel_rank=0,
        device_index=0,
        compute_dtype="torch.float32",
        backend_identity="qwen3-test",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
        local_query_heads=8,
        local_kv_heads=2,
        kv_block_table_width=8,
        proposal_kv_capacity=128,
        blockwise_offload=False,
    )


def _backend(*, live_cache, static_backend=None, broadcast=None):
    return Qwen3DraftCudaGraphBackend(
        backend=(
            _StaticDraftBackend()
            if static_backend is None
            else static_backend
        ),
        proposal_kv_cache=live_cache,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        block_table_width=8,
        torch_module=_FakeTorch(),
        broadcast=(
            _Broadcast()
            if broadcast is None
            else broadcast
        ),
    )


def test_static_allocation_uses_exact_q4_batch4_shapes_and_dtypes():
    live_cache, _ = _caches()
    backend = _backend(live_cache=live_cache)

    tensors = backend._allocate_tensors(_identity())

    assert tensors.first_tokens.shape == (4,)
    assert tensors.current_tokens.shape == (4,)
    assert tensors.positions.shape == (3, 4)
    assert tensors.slot_mapping.shape == (3, 4)
    assert tensors.context_lens.shape == (3, 4)
    assert tensors.block_tables.shape == (3, 4, 8)
    assert tensors.proposal_tokens.shape == (4, 4)
    assert tensors.first_tokens.dtype == torch.int64
    assert tensors.slot_mapping.dtype == torch.int32
    assert tensors.context_lens.dtype == torch.int32
    assert tensors.block_tables.dtype == torch.int32


def test_capture_runs_three_gpu_chained_forward_argmax_broadcast_steps():
    live_cache, scratch_cache = _caches()
    rows = _indexed_rows(live_cache)
    owner = Qwen3DraftGraphScratchOwner(
        live_cache=live_cache,
        scratch_cache=scratch_cache,
    )
    scratch_lease = owner.acquire(_identity(), rows)
    static_backend = _StaticDraftBackend()
    broadcast = _Broadcast()
    backend = _backend(
        live_cache=live_cache,
        static_backend=static_backend,
        broadcast=broadcast,
    )

    entry = backend.capture(
        _identity(),
        scratch_lease.rows,
        eager=lambda *_: None,
        scratch_lease=scratch_lease,
    )

    assert isinstance(entry, AutoregressiveDraftGraphEntry)
    assert len(static_backend.calls) == 6
    assert len(broadcast.calls) == 6
    assert backend.torch.cuda.graph_entries == 1
    captured_calls = static_backend.calls[-3:]
    assert captured_calls[0]["input_ids"].tolist() == [31, 32, 33, 34]
    assert captured_calls[1]["input_ids"].tolist() == [5, 5, 5, 5]
    assert captured_calls[2]["input_ids"].tolist() == [6, 6, 6, 6]
    assert all(src == 0 for _, src in broadcast.calls)
    owner.rollback(scratch_lease)


def test_replay_prepares_live_transactions_reads_tokens_once_and_returns_group():
    live_cache, _ = _caches()
    rows = _indexed_rows(live_cache)
    backend = _backend(live_cache=live_cache)
    tensors = backend._allocate_tensors(_identity())
    proposal_tokens = _TrackingProposalTokens(
        (
            (31, 5, 6, 7),
            (32, 5, 6, 7),
            (33, 5, 6, 7),
            (34, 5, 6, 7),
        )
    )
    tensors.proposal_tokens = proposal_tokens
    graph = _FakeGraph(
        replay_callback=lambda: proposal_tokens.tensor.copy_(
            torch.tensor([
                [31, 5, 6, 7],
                [32, 5, 6, 7],
                [33, 5, 6, 7],
                [34, 5, 6, 7],
            ], dtype=torch.int64)
        )
    )
    entry = AutoregressiveDraftGraphEntry(
        identity=_identity(),
        graph=Qwen3DraftCudaGraphPayload(
            graph=graph,
            tensors=tensors,
        ),
        static_bytes=1,
        capture_duration_ns=1,
        reserved_delta_bytes=1,
    )

    result = backend.replay(entry, rows)

    assert isinstance(result, AutoregressiveDraftGroupExecution)
    assert result.execution_mode == "cuda_graph"
    assert result.token_rows == (
        (31, 5, 6, 7),
        (32, 5, 6, 7),
        (33, 5, 6, 7),
        (34, 5, 6, 7),
    )
    assert len(result.transactions) == 4
    assert all(
        transaction.state == "reserved"
        for transaction in result.transactions
    )
    assert proposal_tokens.tolist_calls == 1
    assert graph.replay_calls == 1


def test_pre_replay_validation_happens_before_graph_entry():
    live_cache, _ = _caches()
    rows = _indexed_rows(live_cache)
    backend = _backend(live_cache=live_cache)
    tensors = backend._allocate_tensors(_identity())
    graph = _FakeGraph()
    entry = AutoregressiveDraftGraphEntry(
        identity=_identity(),
        graph=Qwen3DraftCudaGraphPayload(
            graph=graph,
            tensors=tensors,
        ),
        static_bytes=1,
        capture_duration_ns=1,
        reserved_delta_bytes=1,
    )

    with pytest.raises(
        AutoregressiveDraftGraphPreReplayError,
        match="row count",
    ):
        backend.replay(entry, rows[:3])

    assert graph.replay_calls == 0
    assert live_cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_replay_failure_aborts_every_live_transaction():
    live_cache, _ = _caches()
    rows = _indexed_rows(live_cache)
    backend = _backend(live_cache=live_cache)
    tensors = backend._allocate_tensors(_identity())

    def fail_replay():
        raise RuntimeError("injected replay failure")

    graph = _FakeGraph(replay_callback=fail_replay)
    entry = AutoregressiveDraftGraphEntry(
        identity=_identity(),
        graph=Qwen3DraftCudaGraphPayload(
            graph=graph,
            tensors=tensors,
        ),
        static_bytes=1,
        capture_duration_ns=1,
        reserved_delta_bytes=1,
    )

    with pytest.raises(RuntimeError, match="injected replay failure"):
        backend.replay(entry, rows)

    assert graph.replay_calls == 1
    assert live_cache.authority_snapshot()[
        "active_transaction_count"
    ] == 0


def test_broadcast_failure_during_capture_is_not_silently_ignored():
    live_cache, scratch_cache = _caches()
    rows = _indexed_rows(live_cache)
    owner = Qwen3DraftGraphScratchOwner(
        live_cache=live_cache,
        scratch_cache=scratch_cache,
    )
    scratch_lease = owner.acquire(_identity(), rows)
    backend = _backend(
        live_cache=live_cache,
        broadcast=_Broadcast(fail=True),
    )

    with pytest.raises(
        RuntimeError,
        match="injected broadcast failure",
    ):
        backend.capture(
            _identity(),
            scratch_lease.rows,
            eager=lambda *_: None,
            scratch_lease=scratch_lease,
        )

    owner.rollback(scratch_lease)
