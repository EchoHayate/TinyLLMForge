from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from pathlib import Path
import sys
import types

import numpy as np
import pytest

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

from tinyvllm.engine.qwen35_mtp_graph import (
    Qwen35MTPGraphEntry,
    Qwen35MTPGraphIdentity,
    Qwen35MTPGraphPreReplayError,
)
from tinyvllm.engine.qwen35_mtp_cuda_graph_backend import (
    Qwen35MTPCudaGraphBackend,
    Qwen35MTPCudaGraphPayload,
)


def _shape_of(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _shape_of(data[0])


def _zeros(shape):
    if not shape:
        return 0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class _FakeDType:

    def __init__(self, name, element_size):
        self.name = name
        self.element_size = element_size

    def __str__(self):
        return self.name


class _FakeTensor:

    def __init__(self, data, *, dtype, device):
        numpy_dtype = (
            np.int64
            if dtype.name == "torch.int64"
            else np.int32
            if dtype.name == "torch.int32"
            else np.float32
        )
        self.array = np.array(data, dtype=numpy_dtype)
        self.dtype = dtype
        self.device = device
        self.shape = self.array.shape
        self.fail_copy = False

    def numel(self):
        return int(self.array.size)

    def element_size(self):
        return self.dtype.element_size

    def copy_(self, other):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        if self.shape != other.shape:
            raise RuntimeError("shape mismatch")
        self.array[...] = other.array
        return self

    def tolist(self):
        return self.array.tolist()

    def __getitem__(self, key):
        tensor = object.__new__(_FakeTensor)
        tensor.array = self.array[key]
        tensor.dtype = self.dtype
        tensor.device = self.device
        tensor.shape = tensor.array.shape
        tensor.fail_copy = False
        return tensor


class _FakeTorch:

    int64 = _FakeDType("torch.int64", 8)
    int32 = _FakeDType("torch.int32", 4)
    bfloat16 = _FakeDType("torch.bfloat16", 2)

    @staticmethod
    def zeros(*shape, dtype, device):
        numpy_dtype = (
            np.int64
            if dtype.name == "torch.int64"
            else np.int32
            if dtype.name == "torch.int32"
            else np.float32
        )
        return _FakeTensor(
            np.zeros(
                tuple(int(value) for value in shape),
                dtype=numpy_dtype,
            ),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def tensor(data, *, dtype, device):
        return _FakeTensor(data, dtype=dtype, device=device)

    @staticmethod
    def cat(tensors, dim=0):
        return _FakeTensor(
            np.concatenate(
                tuple(tensor.array for tensor in tensors),
                axis=dim,
            ),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

    @staticmethod
    def argmax(tensor, dim=-1):
        return _FakeTensor(
            np.argmax(tensor.array, axis=dim),
            dtype=_FakeTorch.int64,
            device=tensor.device,
        )


class _FakeCudaGraph:

    def __init__(self):
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1


class _FakeCuda:

    CUDAGraph = _FakeCudaGraph

    def __init__(self):
        self.synchronize_calls = 0
        self.graph_pool = object()

    def graph_pool_handle(self):
        return self.graph_pool

    @staticmethod
    def Stream():
        return object()

    def synchronize(self):
        self.synchronize_calls += 1

    @staticmethod
    def memory_allocated():
        return 100

    @staticmethod
    def memory_reserved():
        return 200

    @staticmethod
    @contextmanager
    def graph(_graph, pool=None, stream=None):
        del pool, stream
        yield


_FakeTorch.cuda = _FakeCuda()


@dataclass(frozen=True)
class _InputRow:
    sequence_id: int
    token_ids: tuple[int, ...]
    first_target_token: int
    target_hidden: _FakeTensor


@dataclass(frozen=True)
class _Bootstrap:
    sequence_epoch: int


@dataclass
class _Transaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    staged_slot_ids: tuple[int, ...]
    state: str = "reserved"


class _ProposalCache:

    def __init__(self, committed):
        self.committed = committed
        self.transactions = {}
        self.begin_calls = []
        self.abort_calls = []
        self.materialized_calls = []

    def committed_slot_ids(self, sequence_id):
        return self.committed[sequence_id]

    def begin(
        self,
        sequence_id,
        sequence_epoch,
        staged_entry_count,
    ):
        self.begin_calls.append((
            sequence_id,
            sequence_epoch,
            staged_entry_count,
        ))
        transaction = _Transaction(
            transaction_id=f"live-{sequence_id}",
            sequence_id=sequence_id,
            sequence_epoch=sequence_epoch,
            staged_slot_ids=tuple(
                range(
                    1000 + 10 * len(self.begin_calls),
                    1000 + 10 * len(self.begin_calls)
                    + staged_entry_count,
                )
            ),
        )
        self.transactions[transaction.transaction_id] = transaction
        return transaction

    def abort(self, transaction_id):
        transaction = self.transactions[transaction_id]
        transaction.state = "aborted"
        self.abort_calls.append(transaction_id)

    def mark_materialized(
        self,
        transaction,
        materialized_entry_count,
    ):
        assert transaction.state == "reserved"
        transaction.state = "materialized"
        self.materialized_calls.append((
            transaction.transaction_id,
            materialized_entry_count,
        ))


class _FakeModule:

    def __init__(self):
        self.forward_calls = []

    def forward_step(self, input_ids, positions, hidden_states):
        self.forward_calls.append((
            input_ids.tolist(),
            positions.tolist(),
            hidden_states.tolist(),
        ))
        next_hidden = _FakeTensor(
            hidden_states.array + 1,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        logits = np.zeros((input_ids.shape[0], 128))
        for row_index, token_id in enumerate(
            input_ids.array.tolist()
        ):
            logits[row_index, int(token_id) + 1] = 1
        return next_hidden, _FakeTensor(
            logits,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


def _identity(*, exact_q=4, exact_batch_size=1, hidden_size=4):
    return Qwen35MTPGraphIdentity(
        exact_q=exact_q,
        exact_batch_size=exact_batch_size,
        device_index=0,
        compute_dtype="torch.bfloat16",
        hidden_size=hidden_size,
        mtp_layer_count=1,
        block_table_width=6,
    )


def _hidden(values):
    return _FakeTorch.tensor(
        [values],
        dtype=_FakeTorch.bfloat16,
        device="cuda:0",
    )


def _rows(batch_size=1):
    source = (
        (
            _InputRow(7, (10, 11, 12), 91, _hidden([1, 2, 3, 4])),
            _Bootstrap(5),
        ),
        (
            _InputRow(9, (20,), 92, _hidden([5, 6, 7, 8])),
            _Bootstrap(6),
        ),
        (
            _InputRow(11, (30, 31), 93, _hidden([9, 10, 11, 12])),
            _Bootstrap(7),
        ),
        (
            _InputRow(13, (40, 41, 42, 43), 94, _hidden([13, 14, 15, 16])),
            _Bootstrap(8),
        ),
    )
    return source[:batch_size]


def _backend(*, committed=None, module=None, context_calls=None):
    if committed is None:
        committed = {
            7: (1, 2),
            9: (3,),
            11: (4, 5),
            13: (),
        }
    cache = _ProposalCache(committed)
    if module is None:
        module = object()
    if context_calls is None:
        context_calls = []

    @contextmanager
    def temporary_context_factory(**changes):
        context_calls.append(changes)
        yield changes

    backend = Qwen35MTPCudaGraphBackend(
        module=module,
        proposal_kv_cache=cache,
        torch_module=_FakeTorch,
        temporary_context_factory=temporary_context_factory,
        device="cuda:0",
        compute_dtype=_FakeTorch.bfloat16,
        hidden_size=4,
        block_table_width=6,
    )
    return backend, cache


@pytest.mark.parametrize(
    ("batch_size", "exact_q"),
    (
        (1, 2),
        (1, 3),
        (1, 4),
        (4, 2),
        (4, 3),
        (4, 4),
    ),
)
def test_static_layout_is_exact_for_q_and_batch(batch_size, exact_q):
    backend, _ = _backend()
    identity = _identity(
        exact_q=exact_q,
        exact_batch_size=batch_size,
    )

    tensors = backend._allocate_tensors(identity)

    steps = exact_q - 1
    assert tensors.first_tokens.shape == (batch_size,)
    assert tensors.current_tokens.shape == (batch_size,)
    assert tensors.positions.shape == (steps, batch_size)
    assert tensors.initial_hidden.shape == (batch_size, 4)
    assert tensors.current_hidden.shape == (batch_size, 4)
    assert tensors.next_hidden.shape == (batch_size, 4)
    assert tensors.slot_mapping.shape == (steps, batch_size)
    assert tensors.context_lens.shape == (steps, batch_size)
    assert tensors.block_tables.shape == (
        steps,
        batch_size,
        6,
    )
    assert tensors.proposal_tokens.shape == (
        batch_size,
        exact_q,
    )
    assert tensors.first_tokens.dtype is _FakeTorch.int64
    assert tensors.slot_mapping.dtype is _FakeTorch.int32
    assert tensors.initial_hidden.dtype is _FakeTorch.bfloat16


def test_static_byte_estimate_matches_allocated_tensors():
    backend, _ = _backend()
    identity = _identity(exact_q=4, exact_batch_size=4)

    tensors = backend._allocate_tensors(identity)
    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors.__dict__.values()
    )

    assert (
        backend.estimate_static_bytes(identity, _rows(4))
        == expected
    )


def test_prepare_live_replay_builds_every_step_metadata():
    backend, cache = _backend()
    identity = _identity(exact_q=4, exact_batch_size=1)
    tensors = backend._allocate_tensors(identity)

    transactions = backend._prepare_live_replay(
        identity,
        tensors,
        _rows(1),
    )

    assert len(transactions) == 1
    assert cache.begin_calls == [(7, 5, 3)]
    assert tensors.first_tokens.tolist() == [91]
    assert tensors.current_tokens.tolist() == [91]
    assert tensors.positions.tolist() == [[2], [3], [4]]
    assert tensors.initial_hidden.tolist() == [[1, 2, 3, 4]]
    assert tensors.current_hidden.tolist() == [[1, 2, 3, 4]]
    assert tensors.slot_mapping.tolist() == [
        [1010],
        [1011],
        [1012],
    ]
    assert tensors.context_lens.tolist() == [[3], [4], [5]]
    assert tensors.block_tables.tolist() == [
        [[1, 2, 1010, 0, 0, 0]],
        [[1, 2, 1010, 1011, 0, 0]],
        [[1, 2, 1010, 1011, 1012, 0]],
    ]
    assert tensors.proposal_tokens.tolist() == [
        [91, 0, 0, 0],
    ]


def test_prepare_live_replay_preserves_batch_row_order():
    backend, cache = _backend()
    identity = _identity(exact_q=2, exact_batch_size=4)
    tensors = backend._allocate_tensors(identity)

    transactions = backend._prepare_live_replay(
        identity,
        tensors,
        _rows(4),
    )

    assert tuple(
        transaction.sequence_id for transaction in transactions
    ) == (7, 9, 11, 13)
    assert tensors.first_tokens.tolist() == [91, 92, 93, 94]
    assert cache.begin_calls == [
        (7, 5, 1),
        (9, 6, 1),
        (11, 7, 1),
        (13, 8, 1),
    ]


def test_prepare_overflow_aborts_every_started_transaction():
    backend, cache = _backend(
        committed={7: (1, 2, 3, 4, 5, 6)}
    )
    identity = _identity(exact_q=2, exact_batch_size=1)
    tensors = backend._allocate_tensors(identity)

    with pytest.raises(
        Qwen35MTPGraphPreReplayError,
        match="block table",
    ):
        backend._prepare_live_replay(
            identity,
            tensors,
            _rows(1),
        )

    assert cache.abort_calls == ["live-7"]


def test_prepare_copy_failure_aborts_every_started_transaction():
    backend, cache = _backend()
    identity = _identity(exact_q=2, exact_batch_size=1)
    tensors = backend._allocate_tensors(identity)
    tensors.first_tokens.fail_copy = True

    with pytest.raises(
        Qwen35MTPGraphPreReplayError,
        match="static input",
    ):
        backend._prepare_live_replay(
            identity,
            tensors,
            _rows(1),
        )

    assert cache.abort_calls == ["live-7"]


def test_prepare_rejects_exact_batch_mismatch_before_transaction():
    backend, cache = _backend()
    identity = _identity(exact_q=2, exact_batch_size=4)
    tensors = backend._allocate_tensors(identity)

    with pytest.raises(
        Qwen35MTPGraphPreReplayError,
        match="batch size",
    ):
        backend._prepare_live_replay(
            identity,
            tensors,
            _rows(1),
        )

    assert cache.begin_calls == []


def test_static_chain_uses_gpu_argmax_for_every_mtp_step():
    module = _FakeModule()
    context_calls = []
    backend, _ = _backend(
        module=module,
        context_calls=context_calls,
    )
    identity = _identity(exact_q=4, exact_batch_size=1)
    tensors = backend._allocate_tensors(identity)
    backend._prepare_live_replay(identity, tensors, _rows(1))

    backend._run_static_chain(identity, tensors)

    assert len(module.forward_calls) == 3
    assert [
        call[0] for call in module.forward_calls
    ] == [[91], [92], [93]]
    assert tensors.proposal_tokens.tolist() == [[91, 92, 93, 94]]
    assert len(context_calls) == 3
    assert all(
        call["mode"] == "decode"
        and call["force_attention_backend"] is True
        and call["kv_offload_manager"] is None
        and call["kv_offload_blockwise_decode"] is False
        for call in context_calls
    )


def test_backend_source_has_no_host_scalar_extraction():
    source = inspect.getsource(Qwen35MTPCudaGraphBackend)
    assert ".item(" not in source
    assert "torch.argmax" in source


def test_exact_graph_families_share_one_private_cuda_pool():
    source = inspect.getsource(Qwen35MTPCudaGraphBackend)

    assert "torch_module.cuda.graph_pool_handle()" in source
    assert "torch_module.cuda.Stream()" in source
    assert "pool=self.graph_pool" in source
    assert "stream=self.capture_stream" in source


def test_capture_returns_real_graph_entry_with_exact_static_bytes():
    module = _FakeModule()
    backend, cache = _backend(module=module)
    identity = _identity(exact_q=3, exact_batch_size=1)
    input_row, bootstrap = _rows(1)[0]
    transaction = cache.begin(
        input_row.sequence_id,
        bootstrap.sequence_epoch,
        2,
    )
    scratch_row = types.SimpleNamespace(
        input_row=input_row,
        bootstrap=bootstrap,
        source_committed_slot_ids=(1, 2),
        transaction=transaction,
    )
    scratch_lease = types.SimpleNamespace(rows=(scratch_row,))

    entry = backend.capture(
        identity,
        scratch_lease.rows,
        eager=lambda *_: pytest.fail(
            "capture must not invoke eager callback"
        ),
        scratch_lease=scratch_lease,
    )

    assert isinstance(entry.graph.graph, _FakeCudaGraph)
    assert entry.identity == identity
    assert entry.static_bytes == backend.estimate_static_bytes(
        identity,
        scratch_lease.rows,
    )
    assert entry.capture_duration_ns >= 0
    assert len(module.forward_calls) >= identity.exact_q - 1


def test_qwen_attention_has_capture_safe_backend_bypass_contract():
    context_source = (
        ROOT / "tinyvllm/utils/context.py"
    ).read_text()
    attention_source = (
        ROOT / "tinyvllm/layers/qwen35_full_attention.py"
    ).read_text()

    assert "force_attention_backend: bool = False" in context_source
    assert (
        'getattr(context, "force_attention_backend", False)'
        in attention_source
    )
    assert "qwen35_cached_decode_graph_attention" in attention_source
    assert ".k_cache.shape[1] == 1" in attention_source
    graph_helper = attention_source.split(
        "def qwen35_cached_decode_graph_attention",
        1,
    )[1].split(
        "class Qwen35FullAttentionShell",
        1,
    )[0]
    assert ".item(" not in graph_helper
    assert ".tolist(" not in graph_helper


class _ReplayGraph:

    def __init__(self, callback=None, error=None):
        self.callback = callback
        self.error = error
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1
        if self.error is not None:
            raise self.error
        if self.callback is not None:
            self.callback()


def _entry(backend, identity, graph):
    tensors = backend._allocate_tensors(identity)
    return Qwen35MTPGraphEntry(
        identity=identity,
        graph=Qwen35MTPCudaGraphPayload(
            graph=graph,
            tensors=tensors,
        ),
        static_bytes=backend.estimate_static_bytes(
            identity,
            _rows(identity.exact_batch_size),
        ),
        capture_duration_ns=1,
        reserved_delta_bytes=0,
    )


def test_replay_materializes_transactions_and_returns_proposals():
    module = _FakeModule()
    backend, cache = _backend(module=module)
    identity = _identity(exact_q=4, exact_batch_size=1)
    entry = _entry(backend, identity, graph=None)
    graph = _ReplayGraph(
        callback=lambda: backend._run_static_chain(
            identity,
            entry.graph.tensors,
        )
    )
    entry.graph.graph = graph

    proposals = backend.replay(entry, _rows(1))

    assert graph.replay_calls == 1
    assert cache.materialized_calls == [("live-7", 3)]
    assert cache.abort_calls == []
    assert len(proposals) == 1
    assert proposals[0].sequence_id == 7
    assert proposals[0].token_ids == (91, 92, 93, 94)
    assert proposals[0].source_type == "native_model_runner"
    assert proposals[0].metadata == {
        "exact_q": 4,
        "staged_entry_count": 3,
        "execution_mode": "cuda_graph",
    }
    assert proposals[0].proposal_transaction_id == "live-7"


def test_replay_preflight_failure_never_launches_graph():
    backend, cache = _backend()
    identity = _identity(exact_q=2, exact_batch_size=4)
    graph = _ReplayGraph()
    entry = _entry(backend, identity, graph)

    with pytest.raises(
        Qwen35MTPGraphPreReplayError,
        match="batch size",
    ):
        backend.replay(entry, _rows(1))

    assert graph.replay_calls == 0
    assert cache.begin_calls == []


def test_replay_started_failure_aborts_transactions_and_propagates():
    backend, cache = _backend()
    identity = _identity(exact_q=3, exact_batch_size=1)
    graph = _ReplayGraph(error=RuntimeError("replay failed"))
    entry = _entry(backend, identity, graph)

    with pytest.raises(RuntimeError, match="replay failed"):
        backend.replay(entry, _rows(1))

    assert graph.replay_calls == 1
    assert cache.abort_calls == ["live-7"]
    assert cache.materialized_calls == []


def test_post_replay_output_failure_is_hard_and_aborts_transaction():
    backend, cache = _backend()
    identity = _identity(exact_q=2, exact_batch_size=1)
    graph = _ReplayGraph()
    entry = _entry(backend, identity, graph)
    invalid_output = _FakeTensor(
        [[91, 92, 93]],
        dtype=_FakeTorch.int64,
        device="cuda:0",
    )
    graph.callback = lambda: setattr(
        entry.graph.tensors,
        "proposal_tokens",
        invalid_output,
    )

    with pytest.raises(RuntimeError, match="output"):
        backend.replay(entry, _rows(1))

    assert graph.replay_calls == 1
    assert cache.abort_calls == ["live-7"]

