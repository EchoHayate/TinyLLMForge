from __future__ import annotations

from dataclasses import replace
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

from tinyvllm.engine.qwen3_draft_backend import (
    Qwen3AutoregressiveDraftBackend,
    Qwen3DraftPhysicalSlotStore,
)
from tinyvllm.engine.qwen3_draft_proposal_kv import (
    Qwen3DraftProposalKVStorage,
)
from tinyvllm.engine.autoregressive_draft_executor import (
    AutoregressiveDraftDecodeRow,
    AutoregressiveDraftPrefillRow,
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
    assert_tensor_free,
)
from tinyvllm.utils.context import get_context


def _model(
    layer_count=3,
    local_query_heads=4,
    local_kv_heads=2,
    head_dim=4,
):
    layers = []
    for _ in range(layer_count):
        backend = SimpleNamespace(
            k_cache=torch.Tensor(),
            v_cache=torch.Tensor(),
            kv_quant_bits=None,
        )
        attention = SimpleNamespace(
            num_heads=local_query_heads,
            num_kv_heads=local_kv_heads,
            head_dim=head_dim,
            attn=backend,
        )
        layers.append(
            SimpleNamespace(self_attn=attention)
        )
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
    )


def test_store_binds_one_block_size_one_slice_per_layer():
    model = _model()

    store = Qwen3DraftPhysicalSlotStore(
        model,
        capacity=7,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert store.key_cache.shape == (3, 7, 1, 2, 4)
    assert store.value_cache.shape == (3, 7, 1, 2, 4)
    assert store.block_size == 1
    assert store.layer_count == 3
    assert store.local_kv_heads == 2
    assert store.head_dim == 4
    for layer_index, layer in enumerate(
        model.model.layers
    ):
        assert layer.self_attn.attn.k_cache.data_ptr() == (
            store.key_cache[layer_index].data_ptr()
        )
        assert layer.self_attn.attn.v_cache.data_ptr() == (
            store.value_cache[layer_index].data_ptr()
        )
        assert layer.self_attn.attn.k_cache.shape == (
            7,
            1,
            2,
            4,
        )
        assert layer.self_attn.attn.kv_quant_bits == 0


def test_reservation_release_and_zeroing_cover_every_layer():
    store = Qwen3DraftPhysicalSlotStore(
        _model(layer_count=2),
        capacity=3,
        dtype=torch.float32,
        device="cpu",
    )

    assert store.reserve_slots(2) == (0, 1)
    store.key_cache[:, 0:2].fill_(3)
    store.value_cache[:, 0:2].fill_(4)
    with pytest.raises(RuntimeError, match="exhausted"):
        store.reserve_slots(2)

    store.release_slots((1,))

    assert torch.count_nonzero(store.key_cache[:, 1]) == 0
    assert torch.count_nonzero(store.value_cache[:, 1]) == 0
    assert torch.count_nonzero(store.key_cache[:, 0]) > 0
    assert store.reserve_slots(1) == (1,)


def test_slot_identity_covers_every_layer():
    store = Qwen3DraftPhysicalSlotStore(
        _model(layer_count=2),
        capacity=2,
        dtype=torch.float32,
        device="cpu",
    )

    identity = store.slot_identity(1)

    assert len(identity) == 2
    assert identity[0] == (
        store.key_cache[0, 1].data_ptr(),
        store.value_cache[0, 1].data_ptr(),
    )
    assert identity[1] == (
        store.key_cache[1, 1].data_ptr(),
        store.value_cache[1, 1].data_ptr(),
    )


def test_mismatched_layer_kv_shape_is_rejected():
    model = _model(layer_count=2)
    model.model.layers[1].self_attn.num_kv_heads = 1

    with pytest.raises(ValueError, match="identical"):
        Qwen3DraftPhysicalSlotStore(
            model,
            capacity=2,
            dtype=torch.float32,
            device="cpu",
        )


def test_foreign_attention_cache_is_rejected():
    model = _model(layer_count=1)
    model.model.layers[0].self_attn.attn.k_cache = (
        torch.zeros(1, 1, 2, 4)
    )

    with pytest.raises(RuntimeError, match="already owns"):
        Qwen3DraftPhysicalSlotStore(
            model,
            capacity=2,
            dtype=torch.float32,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("capacity", "dtype", "message"),
    (
        (0, torch.float32, "capacity"),
        (True, torch.float32, "capacity"),
        (2, "float32", "dtype"),
    ),
)
def test_invalid_constructor_values_fail(
    capacity,
    dtype,
    message,
):
    with pytest.raises(ValueError, match=message):
        Qwen3DraftPhysicalSlotStore(
            _model(),
            capacity=capacity,
            dtype=dtype,
            device="cpu",
        )


def test_stale_or_duplicate_release_is_rejected():
    store = Qwen3DraftPhysicalSlotStore(
        _model(),
        capacity=3,
        dtype=torch.float32,
        device="cpu",
    )
    store.reserve_slots(2)

    with pytest.raises(RuntimeError, match="stale"):
        store.release_slots((1, 1))
    store.release_slots((1,))
    with pytest.raises(RuntimeError, match="stale"):
        store.release_slots((1,))


def test_authority_snapshot_is_tensor_free():
    store = Qwen3DraftPhysicalSlotStore(
        _model(layer_count=2),
        capacity=3,
        dtype=torch.float32,
        device="cpu",
    )
    store.reserve_slots(2)

    snapshot = store.authority_snapshot()

    assert_tensor_free(snapshot, name="physical store snapshot")
    assert snapshot == {
        "capacity": 3,
        "block_size": 1,
        "layer_count": 2,
        "local_kv_heads": 2,
        "head_dim": 4,
        "dtype": "torch.float32",
        "device": "cpu",
        "allocated_slot_count": 2,
        "free_slot_count": 1,
    }


_DEFAULT_LOGITS = object()


class _FakeQwen3:

    def __init__(
        self,
        *,
        malformed_hidden=False,
        malformed_logits=False,
        logits_result=_DEFAULT_LOGITS,
        local_query_heads=4,
        local_kv_heads=2,
    ):
        shape_model = _model(
            layer_count=2,
            local_query_heads=local_query_heads,
            local_kv_heads=local_kv_heads,
        )
        self.model = shape_model.model
        self.forward_calls = 0
        self.compute_logits_calls = 0
        self.calls = []
        self.malformed_hidden = malformed_hidden
        self.malformed_logits = malformed_logits
        self.logits_result = logits_result
        self._parameter = torch.nn.Parameter(
            torch.ones(3, 5, dtype=torch.float32)
        )

    def parameters(self):
        return (self._parameter,)

    @staticmethod
    def _clone(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return value

    def __call__(self, input_ids, positions):
        self.forward_calls += 1
        context = get_context()
        self.calls.append(SimpleNamespace(
            input_ids=input_ids.detach().cpu().clone(),
            positions=positions.detach().cpu().clone(),
            mode=context.mode,
            is_prefill=context.is_prefill,
            slot_mapping=self._clone(context.slot_mapping),
            context_lens=self._clone(context.context_lens),
            block_tables=self._clone(context.block_tables),
            cu_seqlens_q=self._clone(context.cu_seqlens_q),
            cu_seqlens_k=self._clone(context.cu_seqlens_k),
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            kv_offload_manager=context.kv_offload_manager,
            kv_offload_blockwise_decode=(
                context.kv_offload_blockwise_decode
            ),
            kv_offload_blockwise_prefill=(
                context.kv_offload_blockwise_prefill
            ),
            kv_offload_blockwise_blocks=(
                context.kv_offload_blockwise_blocks
            ),
            kv_offload_logical_block_tables=(
                context.kv_offload_logical_block_tables
            ),
            kv_offload_context_lens=(
                context.kv_offload_context_lens
            ),
            kv_offload_write_blocks=(
                context.kv_offload_write_blocks
            ),
        ))
        row_count = int(input_ids.shape[0])
        if self.malformed_hidden:
            row_count += 1
        return torch.arange(
            row_count * 4,
            dtype=torch.float32,
        ).reshape(row_count, 4)

    def compute_logits(self, hidden_states):
        self.compute_logits_calls += 1
        if self.logits_result is not _DEFAULT_LOGITS:
            return self.logits_result
        row_count = int(hidden_states.shape[0])
        if self.malformed_logits:
            return torch.zeros(row_count, 8, dtype=torch.int64)
        logits = torch.full(
            (row_count, 8),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, 3] = 10.0
        return logits


def _backend_fixture(
    *,
    malformed_hidden=False,
    malformed_logits=False,
    logits_result=_DEFAULT_LOGITS,
    rank=0,
    world_size=1,
    local_query_heads=4,
    local_kv_heads=2,
):
    model = _FakeQwen3(
        malformed_hidden=malformed_hidden,
        malformed_logits=malformed_logits,
        logits_result=logits_result,
        local_query_heads=local_query_heads,
        local_kv_heads=local_kv_heads,
    )
    store = Qwen3DraftPhysicalSlotStore(
        model,
        capacity=32,
        dtype=torch.float32,
        device="cpu",
    )
    cache = ProposalKVCache(DirectProposalKVAllocator(store))
    backend = Qwen3AutoregressiveDraftBackend(
        model=model,
        proposal_kv_cache=cache,
        backend_identity="qwen3-test",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
        tensor_parallel_rank=rank,
        tensor_parallel_size=world_size,
    )
    return backend, model, cache


def test_backend_accepts_residency_manager_multilayer_storage():
    model = _FakeQwen3()
    storage = Qwen3DraftProposalKVStorage(
        model,
        logical_capacity=8,
        gpu_capacity=4,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    allocator = ProposalKVResidencyManager(
        storage=storage,
        copy_backend=SynchronousProposalKVCopyBackend(),
    )
    cache = ProposalKVCache(allocator)

    backend = Qwen3AutoregressiveDraftBackend(
        model=model,
        proposal_kv_cache=cache,
        backend_identity="qwen3-residency-test",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
    )

    assert backend.physical_store is storage
    assert backend.proposal_kv_cache.entry_allocator is allocator


def test_residency_backend_snapshot_exposes_movement_and_replay_counters():
    model = _FakeQwen3()
    storage = Qwen3DraftProposalKVStorage(
        model,
        logical_capacity=3,
        gpu_capacity=1,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    allocator = ProposalKVResidencyManager(
        storage=storage,
        copy_backend=SynchronousProposalKVCopyBackend(),
    )
    cache = ProposalKVCache(allocator)
    backend = Qwen3AutoregressiveDraftBackend(
        model=model,
        proposal_kv_cache=cache,
        backend_identity="qwen3-residency-test",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
    )
    first = allocator.reserve_entries(1)
    first_lease = allocator.ensure_writable(first)
    storage.gpu_key_cache[:, 0].fill_(3)
    storage.gpu_value_cache[:, 0].fill_(4)
    allocator.record_write_complete(first_lease)
    allocator.commit_entries(first)
    second = allocator.reserve_entries(1)
    second_lease = allocator.ensure_writable(second)
    allocator.record_write_complete(second_lease)
    allocator.commit_entries(second)
    allocator.retire_entries(second, writeback=False)
    allocator.ensure_readable(first)

    snapshot = backend.authority_snapshot()
    storage_snapshot = snapshot["physical_store"]
    allocator_snapshot = (
        snapshot["proposal_kv_cache"]["entry_allocator"]
    )

    assert storage_snapshot["cpu_backing_allocated"] is True
    assert storage_snapshot["logical_capacity"] == 3
    assert storage_snapshot["gpu_capacity"] == 1
    assert allocator_snapshot["allocator_mode"] == "residency"
    assert allocator_snapshot["logical_entry_capacity"] == 3
    assert allocator_snapshot["gpu_slot_capacity"] == 1
    assert allocator_snapshot["d2h_entry_count"] == 1
    assert allocator_snapshot["h2d_entry_count"] == 1
    assert allocator_snapshot["d2h_bytes"] == storage.entry_nbytes()
    assert allocator_snapshot["h2d_bytes"] == storage.entry_nbytes()
    assert allocator_snapshot["accepted_entry_copy_count"] == 0
    assert allocator_snapshot["accepted_entry_replay_count"] == 0
    assert (
        allocator_snapshot[
            "accepted_entry_rematerialization_count"
        ]
        == 0
    )
    assert_tensor_free(
        snapshot,
        name="residency backend authority snapshot",
    )


def _prefill_row(cache, transaction, token_ids, positions):
    lease = cache.entry_allocator.ensure_writable(
        transaction.staged_entry_identities
    )
    return AutoregressiveDraftPrefillRow(
        transaction=transaction,
        token_ids=tuple(token_ids),
        positions=tuple(positions),
        physical_slot_ids=lease.physical_slot_ids,
    )


def _complete_prefill(cache, transaction):
    lease = cache.entry_allocator.ensure_writable(
        transaction.staged_entry_identities
    )
    cache.entry_allocator.record_write_complete(lease)


def _decode_row(
    cache,
    transaction,
    *,
    step,
    input_token_id,
    position,
):
    read_identities = (
        cache.committed_entry_identities(transaction.sequence_id)
        + transaction.staged_entry_identities[:step]
    )
    read_lease = cache.entry_allocator.ensure_readable(
        read_identities
    )
    write_lease = cache.entry_allocator.ensure_writable((
        transaction.staged_entry_identities[step],
    ))
    return AutoregressiveDraftDecodeRow(
        transaction=transaction,
        step=step,
        input_token_id=input_token_id,
        position=position,
        writable_physical_slot_id=(
            write_lease.physical_slot_ids[0]
        ),
        visible_physical_slot_ids=(
            read_lease.physical_slot_ids
            + write_lease.physical_slot_ids
        ),
    )


def test_prefill_batch_packs_rows_into_one_model_forward():
    backend, model, cache = _backend_fixture()
    first = cache.begin(1, 0, 2)
    second = cache.begin(2, 0, 3)

    backend.prefill_batch((
        _prefill_row(cache, first, (7, 8), (0, 1)),
        _prefill_row(cache, second, (4, 5, 6), (0, 1, 2)),
    ))

    assert model.forward_calls == 1
    call = model.calls[0]
    assert call.input_ids.tolist() == [7, 8, 4, 5, 6]
    assert call.positions.tolist() == [0, 1, 0, 1, 2]
    assert call.mode == "prefill"
    assert call.is_prefill is True
    assert call.slot_mapping.tolist() == [0, 1, 2, 3, 4]
    assert call.cu_seqlens_q.tolist() == [0, 2, 5]
    assert call.cu_seqlens_k.tolist() == [0, 2, 5]
    assert call.max_seqlen_q == 3
    assert call.max_seqlen_k == 3
    assert call.block_tables is None
    assert call.kv_offload_manager is None
    assert call.kv_offload_blockwise_prefill is False


def test_prefill_rejects_physical_slot_count_mismatch():
    backend, _, cache = _backend_fixture()
    transaction = cache.begin(1, 0, 2)
    row = _prefill_row(
        cache,
        transaction,
        (7, 8),
        (0, 1),
    )

    with pytest.raises(ValueError, match="physical slot count"):
        backend.prefill_batch((
            replace(row, physical_slot_ids=(0,)),
        ))


def _commit_prompt(cache, sequence_id, token_count):
    transaction = cache.begin(sequence_id, 0, token_count)
    _complete_prefill(cache, transaction)
    cache.mark_materialized(transaction, token_count)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=token_count + 1,
    )
    cache.commit_finalize(ticket.ticket_id)


def _decode_rows(cache, sequence_ids):
    rows = []
    for sequence_id in sequence_ids:
        _commit_prompt(cache, sequence_id, 1)
        transaction = cache.begin(sequence_id, 0, 1)
        rows.append(
            _decode_row(
                cache,
                transaction,
                step=0,
                input_token_id=sequence_id + 2,
                position=1,
            )
        )
    return tuple(rows)


def test_tp4_root_backend_returns_full_logit_rows():
    root_logits = torch.tensor([
        [0.0, 4.0, 1.0],
        [3.0, 2.0, 1.0],
    ])
    backend, _, cache = _backend_fixture(
        rank=0,
        world_size=4,
        logits_result=root_logits,
    )

    rows = backend.decode_step_batch(
        _decode_rows(cache, (1, 2))
    )

    assert isinstance(rows, tuple)
    assert len(rows) == 2
    assert all(row.shape == (3,) for row in rows)
    assert torch.equal(torch.stack(rows), root_logits)


def test_tp4_non_root_backend_requires_none_logits():
    backend, model, cache = _backend_fixture(
        rank=2,
        world_size=4,
        logits_result=None,
    )

    rows = backend.decode_step_batch(
        _decode_rows(cache, (1,))
    )

    assert rows is None
    assert model.forward_calls == 1
    assert model.compute_logits_calls == 1


def test_tp4_non_root_local_vocabulary_logits_fail_closed():
    backend, _, cache = _backend_fixture(
        rank=1,
        world_size=4,
        logits_result=torch.ones(1, 8),
    )

    with pytest.raises(
        ValueError,
        match="non-root logits must be None",
    ):
        backend.decode_step_batch(
            _decode_rows(cache, (1,))
        )


@pytest.mark.parametrize(
    ("logits", "message"),
    (
        (None, "root model logits"),
        (torch.ones(1, 8, dtype=torch.int64), "floating"),
        (torch.ones(2, 8), "exact shape"),
        (torch.ones(1, 1), "exact shape"),
        (
            torch.tensor([[0.0, float("nan")]]),
            "finite",
        ),
    ),
)
def test_tp4_root_rejects_malformed_full_logits(logits, message):
    backend, _, cache = _backend_fixture(
        rank=0,
        world_size=4,
        logits_result=logits,
    )

    with pytest.raises(ValueError, match=message):
        backend.decode_step_batch(
            _decode_rows(cache, (1,))
        )


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    (
        (0, 2, "TP1 or TP4"),
        (0, 3, "TP1 or TP4"),
        (0, 5, "TP1 or TP4"),
        (4, 4, "rank"),
    ),
)
def test_backend_rejects_invalid_tensor_parallel_topology(
    rank,
    world_size,
    message,
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _backend_fixture(rank=rank, world_size=world_size)


def test_decode_batch_uses_visible_slots_and_one_model_forward():
    backend, model, cache = _backend_fixture()
    _commit_prompt(cache, 1, 2)
    _commit_prompt(cache, 2, 3)
    first = cache.begin(1, 0, 2)
    second = cache.begin(2, 0, 2)

    rows = backend.decode_step_batch((
        _decode_row(
            cache,
            first,
            step=0,
            input_token_id=7,
            position=2,
        ),
        _decode_row(
            cache,
            second,
            step=0,
            input_token_id=4,
            position=3,
        ),
    ))

    assert model.forward_calls == 1
    assert model.compute_logits_calls == 1
    assert len(rows) == 2
    assert all(row.shape == (8,) for row in rows)
    call = model.calls[0]
    assert call.mode == "decode"
    assert call.is_prefill is False
    assert call.input_ids.tolist() == [7, 4]
    assert call.positions.tolist() == [2, 3]
    assert call.slot_mapping.tolist() == [5, 7]
    expected_first = (0, 1, 5)
    expected_second = (2, 3, 4, 7)
    assert call.block_tables[0, :len(expected_first)].tolist() == (
        list(expected_first)
    )
    assert call.block_tables[1, :len(expected_second)].tolist() == (
        list(expected_second)
    )
    assert call.block_tables[0, len(expected_first):].tolist() == [
        -1
    ]
    assert call.context_lens.tolist() == [3, 4]
    assert call.max_seqlen_q == 1
    assert call.max_seqlen_k == 4
    assert call.kv_offload_manager is None
    assert call.kv_offload_blockwise_decode is False


def test_residency_decode_uses_blockwise_logical_context_for_batch_four():
    model = _FakeQwen3()
    storage = Qwen3DraftProposalKVStorage(
        model,
        logical_capacity=32,
        gpu_capacity=8,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    allocator = ProposalKVResidencyManager(
        storage=storage,
        copy_backend=SynchronousProposalKVCopyBackend(),
    )
    cache = ProposalKVCache(allocator)
    backend = Qwen3AutoregressiveDraftBackend(
        model=model,
        proposal_kv_cache=cache,
        backend_identity="qwen3-residency-test",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
    )
    rows = []
    expected_logical_rows = []
    expected_write_blocks = []
    for sequence_id in range(1, 5):
        _commit_prompt(cache, sequence_id, 3)
        transaction = cache.begin(sequence_id, 0, 1)
        write_identity = transaction.staged_entry_identities[0]
        write_lease = allocator.ensure_writable((write_identity,))
        visible_identities = (
            cache.committed_entry_identities(sequence_id)
            + (write_identity,)
        )
        logical_row = tuple(
            identity.logical_entry_id
            for identity in visible_identities
        )
        rows.append(AutoregressiveDraftDecodeRow(
            transaction=transaction,
            step=0,
            input_token_id=sequence_id + 10,
            position=3,
            writable_physical_slot_id=(
                write_lease.physical_slot_ids[0]
            ),
            visible_physical_slot_ids=(
                write_lease.physical_slot_ids
            ),
            visible_logical_entry_ids=logical_row,
            blockwise_offload=True,
        ))
        expected_logical_rows.append(list(logical_row))
        expected_write_blocks.append(
            write_identity.logical_entry_id
        )

    logits = backend.decode_step_batch(tuple(rows))

    assert len(logits) == 4
    assert model.forward_calls == 1
    call = model.calls[0]
    assert call.block_tables is None
    assert call.context_lens.tolist() == [4, 4, 4, 4]
    assert call.kv_offload_manager is (
        allocator.blockwise_attention_adapter
    )
    assert call.kv_offload_blockwise_decode is True
    assert call.kv_offload_blockwise_blocks == 1
    assert call.kv_offload_logical_block_tables == (
        expected_logical_rows
    )
    assert call.kv_offload_context_lens == [4, 4, 4, 4]
    assert call.kv_offload_write_blocks == expected_write_blocks


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda row: replace(
                row,
                writable_physical_slot_id=99,
            ),
            "final visible slot",
        ),
        (
            lambda row: replace(
                row,
                visible_physical_slot_ids=(0, 0),
            ),
            "unique",
        ),
    ),
)
def test_decode_rejects_invalid_ephemeral_physical_mapping(
    mutate,
    message,
):
    backend, _, cache = _backend_fixture()
    _commit_prompt(cache, 1, 1)
    transaction = cache.begin(1, 0, 1)
    row = _decode_row(
        cache,
        transaction,
        step=0,
        input_token_id=4,
        position=1,
    )

    with pytest.raises(ValueError, match=message):
        backend.decode_step_batch((mutate(row),))


def test_backend_authority_snapshot_counts_real_model_forwards():
    backend, model, cache = _backend_fixture()
    prompt = cache.begin(1, 0, 2)
    backend.prefill_batch((
        _prefill_row(cache, prompt, (7, 8), (0, 1)),
    ))
    _complete_prefill(cache, prompt)
    cache.mark_materialized(prompt, 2)
    ticket = cache.prepare_finalize(
        prompt.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(ticket.ticket_id)
    decode = cache.begin(1, 0, 1)
    backend.decode_step_batch((
        _decode_row(
            cache,
            decode,
            step=0,
            input_token_id=9,
            position=2,
        ),
    ))

    snapshot = backend.authority_snapshot()

    assert model.forward_calls == 2
    assert snapshot["prefill_forward_count"] == 1
    assert snapshot["decode_forward_count"] == 1
    assert snapshot["real_draft_forward_count"] == 2
    assert snapshot["physical_store"]["capacity"] == 32
    assert_tensor_free(snapshot, name="Qwen3 backend snapshot")


def test_backend_snapshot_reports_rank_local_geometry_and_bytes():
    backend, model, _ = _backend_fixture(
        rank=3,
        world_size=4,
        local_query_heads=8,
        local_kv_heads=2,
    )

    snapshot = backend.authority_snapshot()

    assert snapshot["tensor_parallel_rank"] == 3
    assert snapshot["tensor_parallel_size"] == 4
    assert snapshot["local_query_heads"] == 8
    assert snapshot["local_kv_heads"] == 2
    assert snapshot["local_model_parameter_bytes"] == sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    assert snapshot["local_proposal_kv_bytes"] > 0
    assert snapshot["local_prefill_forward_count"] == 0
    assert snapshot["local_decode_forward_count"] == 0
    assert_tensor_free(snapshot, name="rank-local backend snapshot")


def test_prefill_hidden_row_mismatch_is_rejected():
    backend, _, cache = _backend_fixture(
        malformed_hidden=True,
    )
    transaction = cache.begin(1, 0, 2)

    with pytest.raises(ValueError, match="hidden row count"):
        backend.prefill_batch((
            _prefill_row(
                cache,
                transaction,
                (2, 3),
                (0, 1),
            ),
        ))


def test_decode_logits_must_be_floating_and_exact_shape():
    backend, _, cache = _backend_fixture(
        malformed_logits=True,
    )
    _commit_prompt(cache, 1, 1)
    transaction = cache.begin(1, 0, 1)

    with pytest.raises(ValueError, match="floating"):
        backend.decode_step_batch((
            _decode_row(
                cache,
                transaction,
                step=0,
                input_token_id=2,
                position=1,
            ),
        ))
