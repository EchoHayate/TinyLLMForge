from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

hybrid_module = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)
side_state_module = _load_module(
    "tinyvllm.engine.speculative_side_state",
    "tinyvllm/engine/speculative_side_state.py",
)
trace_module = _load_module(
    "tinyvllm.engine.qwen35_speculative_trace",
    "tinyvllm/engine/qwen35_speculative_trace.py",
)
qwen35_side_state_module = _load_module(
    "tinyvllm.engine.qwen35_speculative_state",
    "tinyvllm/engine/qwen35_speculative_state.py",
)

HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateLease = hybrid_module.HybridStateLease
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
SpeculativeSideStateSelectionRow = (
    side_state_module.SpeculativeSideStateSelectionRow
)
Qwen35SpeculativeStateOwner = (
    qwen35_side_state_module.Qwen35SpeculativeStateOwner
)
Qwen35SpeculativeTraceRecorder = (
    trace_module.Qwen35SpeculativeTraceRecorder
)
fingerprint_candidate_inventory = (
    trace_module.fingerprint_candidate_inventory
)


class _Sequence:
    def __init__(self, sequence_id):
        self.seq_id = sequence_id


def _fixture(batch_size=4):
    components = []
    for layer_index in (1, 3):
        components.extend((
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (2, 2),
                torch.float32,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 2, 2),
                torch.float32,
            ),
        ))
    pool = HybridStateTensorPool(
        HybridStateLayout(tuple(components)),
        batch_size,
        "cpu",
    )
    leases = tuple(
        HybridStateLease(index, 1, 100 + index)
        for index in range(batch_size)
    )
    sequences = tuple(
        _Sequence(1000 + index)
        for index in range(batch_size)
    )
    for lease in leases:
        pool.activate(lease)
    adapters = tuple(
        Qwen35LayerStateAdapter(pool, layer_index)
        for layer_index in (1, 3)
    )
    for layer_offset, adapter in enumerate(adapters):
        for slot_id in range(batch_size):
            adapter.convolution[slot_id].copy_(
                torch.arange(4, dtype=torch.float32).reshape(2, 2)
                + layer_offset * 1000
                + slot_id * 100
            )
            adapter.recurrent[slot_id].copy_(
                torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
                + layer_offset * 2000
                + slot_id * 200
            )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    owner = Qwen35SpeculativeStateOwner(transaction)
    return pool, sequences, leases, transaction, owner


def _clone_candidates(candidates):
    return tuple(
        (
            convolution.clone(),
            recurrent.clone(),
        )
        for convolution, recurrent in candidates
    )


def _offset_candidates(candidates, offset):
    return tuple(
        (
            convolution + offset,
            recurrent + offset * 2,
        )
        for convolution, recurrent in candidates
    )


def _prepared_step(
    leases,
    final_candidates,
    *,
    prefix_candidates=None,
):
    return types.SimpleNamespace(
        leases=leases,
        token_counts=tuple(
            1
            if prefix_candidates is None
            else len(prefix_candidates[index])
            for index in range(len(leases))
        ),
        final_candidates=final_candidates,
        prefix_candidates=prefix_candidates,
    )


def _tail_prefixes(first_candidates, prefix_count=3):
    batch_size = first_candidates[0][0].shape[0]
    return tuple(
        tuple(
            tuple(
                (
                    convolution[sequence_index]
                    + (prefix_index + 1) * 10
                    + sequence_index,
                    recurrent[sequence_index]
                    + (prefix_index + 1) * 20
                    + sequence_index,
                )
                for convolution, recurrent in first_candidates
            )
            for prefix_index in range(prefix_count)
        )
        for sequence_index in range(batch_size)
    )


def _batched_final_from_prefixes(prefixes):
    layer_count = len(prefixes[0][-1])
    return tuple(
        (
            torch.stack(tuple(
                sequence_prefixes[-1][layer_index][0]
                for sequence_prefixes in prefixes
            )),
            torch.stack(tuple(
                sequence_prefixes[-1][layer_index][1]
                for sequence_prefixes in prefixes
            )),
        )
        for layer_index in range(layer_count)
    )


def _selection_row(sequence_id, committed_input_count):
    return SpeculativeSideStateSelectionRow(
        sequence_id=sequence_id,
        proposal_token_count=4,
        accepted_draft_count=committed_input_count - 1,
        verify_input_count=3,
        committed_tail_input_count=committed_input_count - 1,
        committed_input_count=committed_input_count,
    )


def _assert_candidates_equal(actual, expected):
    for actual_pair, expected_pair in zip(actual, expected):
        torch.testing.assert_close(actual_pair[0], expected_pair[0])
        torch.testing.assert_close(actual_pair[1], expected_pair[1])


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(
            _contains_tensor(key) or _contains_tensor(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_candidate_fingerprint_is_clone_stable_and_value_sensitive():
    candidates = (
        (
            torch.tensor(
                [[1.0, 2.0]],
                dtype=torch.bfloat16,
            ),
            torch.tensor(
                [[3.0, 4.0]],
                dtype=torch.float32,
            ),
        ),
    )
    cloned = tuple(
        (convolution.clone(), recurrent.clone())
        for convolution, recurrent in candidates
    )
    changed = (
        (
            candidates[0][0].clone(),
            candidates[0][1].clone(),
        ),
    )
    changed[0][1][0, 0] = 9.0

    assert (
        fingerprint_candidate_inventory(candidates)
        == fingerprint_candidate_inventory(cloned)
    )
    assert (
        fingerprint_candidate_inventory(candidates)
        != fingerprint_candidate_inventory(changed)
    )


def test_side_state_trace_is_default_off():
    _, _, _, _, owner = _fixture(batch_size=1)
    assert owner.drain_trace_rows() == ()


def test_side_state_trace_records_first_tail_and_selection():
    _, sequences, leases, transaction, owner = _fixture(
        batch_size=1
    )
    owner.enable_trace_recording(True)
    handle = owner.prepare(sequences, leases)
    original = _clone_candidates(transaction.gather(leases))
    first_candidates = _offset_candidates(original, 5)
    owner.record_first_target(
        _prepared_step(leases, first_candidates)
    )
    prefixes = _tail_prefixes(first_candidates)
    owner.record_tail(
        _prepared_step(
            leases,
            _batched_final_from_prefixes(prefixes),
            prefix_candidates=prefixes,
        ),
        tuple(sequence.seq_id for sequence in sequences),
    )
    owner.select(
        handle,
        (
            _selection_row(
                sequences[0].seq_id,
                3,
            ),
        ),
    )
    rows = owner.drain_trace_rows()

    assert rows[0]["event"] == "first_target_checkpoint"
    assert rows[0]["checkpoint_index"] == 1
    assert [
        row["checkpoint_index"]
        for row in rows
        if row["event"] == "tail_checkpoint"
    ] == [2, 3, 4]
    assert rows[-1]["event"] == "selected_checkpoint"
    assert rows[-1]["checkpoint_index"] == 3
    assert rows[-1]["committed_input_count"] == 3


def test_side_state_trace_is_read_only_and_raw_owner_scoped():
    _, sequences, leases, transaction, owner = _fixture(
        batch_size=1
    )
    owner.enable_trace_recording(True)
    handle = owner.prepare(sequences, leases)
    original = _clone_candidates(transaction.gather(leases))
    first_candidates = _offset_candidates(original, 5)
    owner.record_first_target(
        _prepared_step(leases, first_candidates)
    )
    prefixes = _tail_prefixes(first_candidates)
    owner.record_tail(
        _prepared_step(
            leases,
            _batched_final_from_prefixes(prefixes),
            prefix_candidates=prefixes,
        ),
        tuple(sequence.seq_id for sequence in sequences),
    )
    checkpoints_before = {
        sequence_id: {
            checkpoint_index: _clone_candidates(candidates)
            for checkpoint_index, candidates in checkpoints.items()
        }
        for sequence_id, checkpoints in (
            owner._active.checkpoints.items()
        )
    }
    owner.select(
        handle,
        (
            _selection_row(
                sequences[0].seq_id,
                3,
            ),
        ),
    )

    rows = owner.drain_trace_rows()

    for sequence_id, checkpoints in checkpoints_before.items():
        for checkpoint_index, expected in checkpoints.items():
            actual = owner._active.checkpoints[
                sequence_id
            ][checkpoint_index]
            for actual_pair, expected_pair in zip(
                actual,
                expected,
            ):
                assert torch.equal(
                    actual_pair[0],
                    expected_pair[0],
                )
                assert torch.equal(
                    actual_pair[1],
                    expected_pair[1],
                )
    assert all(
        set(row) == {
            "sequence_id",
            "event",
            "checkpoint_index",
            "committed_input_count",
            "fingerprint",
        }
        for row in rows
    )
    assert all(
        not any(
            isinstance(value, torch.Tensor)
            for value in row.values()
        )
        for row in rows
    )
    worker_owned_fields = {
        "schema",
        "policy",
        "batch_size",
        "engine_step",
        "proposal_tokens",
        "accepted_tokens",
        "verify_count",
        "fallback",
    }
    assert all(
        worker_owned_fields.isdisjoint(row)
        for row in rows
    )


def test_lifecycle_selects_partial_prefix_and_rollback_restores_original():
    _, sequences, leases, transaction, owner = _fixture()
    original = _clone_candidates(transaction.gather(leases))

    handle = owner.prepare(sequences, leases)
    assert handle == {
        "operation": "prepare",
        "status": "prepared",
        "transaction_id": handle["transaction_id"],
        "sequence_ids": [sequence.seq_id for sequence in sequences],
    }
    assert not _contains_tensor(handle)
    with pytest.raises(RuntimeError, match="active"):
        owner.prepare(sequences, leases)

    first_candidates = _offset_candidates(original, 5)
    first_receipt = owner.record_first_target(
        _prepared_step(leases, first_candidates)
    )
    assert first_receipt["status"] == "recorded"
    assert not _contains_tensor(first_receipt)
    _assert_candidates_equal(
        owner.initial_tail_candidates(
            tuple(sequence.seq_id for sequence in sequences)
        ),
        first_candidates,
    )

    prefixes = _tail_prefixes(first_candidates)
    tail_receipt = owner.record_tail(
        _prepared_step(
            leases,
            _batched_final_from_prefixes(prefixes),
            prefix_candidates=prefixes,
        ),
        tuple(sequence.seq_id for sequence in sequences),
    )
    assert tail_receipt["checkpoint_indices"] == [2, 3, 4]
    assert not _contains_tensor(tail_receipt)

    rows = tuple(
        _selection_row(sequence.seq_id, checkpoint_index)
        for sequence, checkpoint_index in zip(
            sequences,
            (1, 2, 3, 4),
        )
    )
    select_receipt = owner.select(handle, rows)
    assert select_receipt["status"] == "selected"
    assert select_receipt["rows"] == [
        {
            "sequence_id": sequence.seq_id,
            "committed_input_count": checkpoint_index,
            "checkpoint_index": checkpoint_index,
        }
        for sequence, checkpoint_index in zip(
            sequences,
            (1, 2, 3, 4),
        )
    ]
    assert not _contains_tensor(select_receipt)

    apply_receipt = owner.apply(handle)
    assert apply_receipt["status"] == "applied"
    assert not _contains_tensor(apply_receipt)
    applied = transaction.gather(leases)
    for layer_index, applied_pair in enumerate(applied):
        expected_convolution = torch.stack((
            first_candidates[layer_index][0][0],
            prefixes[1][0][layer_index][0],
            prefixes[2][1][layer_index][0],
            prefixes[3][2][layer_index][0],
        ))
        expected_recurrent = torch.stack((
            first_candidates[layer_index][1][0],
            prefixes[1][0][layer_index][1],
            prefixes[2][1][layer_index][1],
            prefixes[3][2][layer_index][1],
        ))
        torch.testing.assert_close(
            applied_pair[0],
            expected_convolution,
        )
        torch.testing.assert_close(
            applied_pair[1],
            expected_recurrent,
        )

    rollback_receipt = owner.rollback(handle)
    assert rollback_receipt["status"] == "rolled_back"
    assert not _contains_tensor(rollback_receipt)
    _assert_candidates_equal(transaction.gather(leases), original)


def test_seal_discards_transaction_and_forbids_rollback():
    _, sequences, leases, transaction, owner = _fixture(batch_size=1)
    original = transaction.gather(leases)
    handle = owner.prepare(sequences, leases)
    first_candidates = _offset_candidates(original, 7)
    owner.record_first_target(
        _prepared_step(leases, first_candidates)
    )
    owner.select(
        handle,
        (_selection_row(sequences[0].seq_id, 1),),
    )
    owner.apply(handle)

    receipt = owner.seal(handle)

    assert receipt["status"] == "sealed"
    assert not _contains_tensor(receipt)
    _assert_candidates_equal(
        transaction.gather(leases),
        first_candidates,
    )
    with pytest.raises(RuntimeError, match="sealed"):
        owner.rollback(handle)
    replacement = owner.prepare(sequences, leases)
    assert replacement["transaction_id"] != handle["transaction_id"]


def test_missing_checkpoint_and_identity_drift_fail_before_apply():
    _, sequences, leases, transaction, owner = _fixture(batch_size=2)
    original = _clone_candidates(transaction.gather(leases))
    handle = owner.prepare(sequences, leases)
    first_candidates = _offset_candidates(original, 3)

    with pytest.raises(ValueError, match="lease identity"):
        owner.record_first_target(
            _prepared_step(
                (leases[1], leases[0]),
                first_candidates,
            )
        )
    owner.record_first_target(
        _prepared_step(leases, first_candidates)
    )
    with pytest.raises(ValueError, match="sequence inventory"):
        owner.initial_tail_candidates(
            (sequences[1].seq_id, sequences[0].seq_id)
        )
    with pytest.raises(ValueError, match="checkpoint"):
        owner.select(
            handle,
            tuple(
                _selection_row(sequence.seq_id, 2)
                for sequence in sequences
            ),
        )
    _assert_candidates_equal(transaction.gather(leases), original)

    stale_handle = dict(handle)
    stale_handle["transaction_id"] = "wrong"
    with pytest.raises(ValueError, match="transaction"):
        owner.apply(stale_handle)
    _assert_candidates_equal(transaction.gather(leases), original)
