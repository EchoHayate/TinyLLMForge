from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

config_module = types.ModuleType("tinyvllm.config")
config_module.Config = object
sys.modules["tinyvllm.config"] = config_module


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_module = types.ModuleType("xxhash")
xxhash_module.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_module)

_load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
_load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
_load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)


class _TorchDType:
    def __init__(self, name, itemsize):
        self.name = name
        self.itemsize = itemsize

    def __str__(self):
        return f"torch.{self.name}"


torch_module = types.ModuleType("torch")
torch_module.float16 = _TorchDType("float16", 2)
torch_module.bfloat16 = _TorchDType("bfloat16", 2)
torch_module.float32 = _TorchDType("float32", 4)
sys.modules.setdefault("torch", torch_module)
hybrid_state_module = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
_HybridStateLease = hybrid_state_module.HybridStateLease
_HybridStateSlotAllocator = (
    hybrid_state_module.HybridStateSlotAllocator
)
_load_module(
    "tinyvllm.engine.speculative_selection",
    "tinyvllm/engine/speculative_selection.py",
)
scheduler_module = _load_module(
    "tinyvllm.engine.scheduler",
    "tinyvllm/engine/scheduler.py",
)

SamplingParams = sys.modules[
    "tinyvllm.sampling_params"
].SamplingParams
Sequence = sys.modules[
    "tinyvllm.engine.sequence"
].Sequence
SequenceStatus = sys.modules[
    "tinyvllm.engine.sequence"
].SequenceStatus
PreparedSchedulerPostprocess = (
    scheduler_module.PreparedSchedulerPostprocess
)
ScheduledOutputRow = scheduler_module.ScheduledOutputRow
Scheduler = scheduler_module.Scheduler
Sequence.block_size = 2


class _IndexableNonIterableBlocks:
    def __init__(self, values):
        self._values = list(values)
        self.index_reads = []

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("block slices are not allowed")
        self.index_reads.append(index)
        return self._values[index]

    def __iter__(self):
        raise AssertionError("full block iteration is not allowed")


class _NonSearchableFreeBlocks(deque):
    def __contains__(self, _value):
        raise AssertionError("free block membership scans are not allowed")


def _config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        max_num_prefill_tokens_per_step=0,
        chunked_prefill_decode_first=True,
        chunked_prefill_max_consecutive_chunks=0,
        chunked_prefill_mixed_batch=False,
        chunked_prefill_mixed_min_prompt_tokens=0,
        chunked_prefill_adaptive_mixed=False,
        chunked_prefill_adaptive_enter_waiting=8,
        chunked_prefill_adaptive_exit_waiting=2,
        chunked_prefill_adaptive_transition_steps=2,
        chunked_prefill_adaptive_max_mixed_steps=2,
        chunked_prefill_slo_mixed=False,
        chunked_prefill_slo_target_gap_ns=0,
        chunked_prefill_slo_reserve_ns=0,
        chunked_prefill_slo_cost_intercept_ns=0,
        chunked_prefill_slo_cost_per_prefill_token_ns=0,
        chunked_prefill_slo_min_chunk_tokens=1,
        eos=99,
        num_kvcache_blocks=32,
        kvcache_block_size=2,
    )


def _running_sequence(
    scheduler,
    prompt,
    *,
    max_tokens=8,
    temperature=0.0,
    ignore_eos=False,
):
    sequence = Sequence(
        list(prompt),
        SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        ),
    )
    scheduler.block_manager.allocate(sequence)
    sequence.num_computed_tokens = len(sequence)
    sequence.status = SequenceStatus.RUNNING
    scheduler.running.append(sequence)
    return sequence


def _scheduled_prefill_sequence(
    scheduler,
    prompt,
    *,
    chunk_end,
    final,
    do_sample,
    max_tokens=8,
):
    sequence = Sequence(
        list(prompt),
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
        ),
    )
    scheduler.block_manager.allocate(
        sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    sequence.num_computed_tokens = 0
    sequence.prefill_chunk_start = 0
    sequence.prefill_chunk_end = chunk_end
    sequence.prefill_chunk_final = final
    sequence.step_is_decode = False
    sequence.step_do_sample = do_sample
    sequence.status = SequenceStatus.WAITING
    return sequence


def _snapshot(scheduler, sequences):
    snapshot = {
        "tokens": {
            sequence.seq_id: tuple(sequence.token_ids)
            for sequence in sequences
        },
        "statuses": {
            sequence.seq_id: sequence.status
            for sequence in sequences
        },
        "blocks": {
            sequence.seq_id: tuple(sequence.block_table)
            for sequence in sequences
        },
        "running": tuple(
            sequence.seq_id for sequence in scheduler.running
        ),
        "free": tuple(scheduler.block_manager.free_block_ids),
        "used": frozenset(
            scheduler.block_manager.used_block_ids
        ),
        "block_metadata": tuple(
            (
                block.ref_count,
                block.generation,
                block.hash,
                tuple(block.token_ids),
            )
            for block in scheduler.block_manager.blocks
        ),
        "hash_to_block_id": dict(
            scheduler.block_manager.hash_to_block_id
        ),
        "hash_to_block_ids": {
            block_hash: frozenset(block_ids)
            for block_hash, block_ids
            in scheduler.block_manager.hash_to_block_ids.items()
        },
        "release_events": tuple(
            scheduler._hybrid_state_release_events
        ),
        "progress": dict(
            scheduler.decode_progress_ns_by_seq_id
        ),
        "slo": dict(scheduler._last_slo_postprocess),
        "prefill_notified": frozenset(
            scheduler._prefill_commit_notified_request_ids
        ),
        "prefill_hook_error": (
            scheduler._prefill_commit_hook_error
        ),
    }
    allocator = scheduler.hybrid_state_allocator
    if allocator is not None:
        snapshot["hybrid"] = {
            "free_slots": tuple(allocator._free_slots),
            "generations": tuple(allocator._generations),
            "owners": dict(allocator._owners),
            "request_leases": dict(
                allocator._request_leases
            ),
        }
    return snapshot


def test_scheduler_exposes_prepared_postprocess_api():
    assert hasattr(scheduler_module, "ScheduledOutputRow")
    assert hasattr(
        scheduler_module,
        "PreparedSchedulerPostprocess",
    )
    assert callable(
        getattr(
            scheduler_module.Scheduler,
            "prepare_postprocess",
            None,
        )
    )


def test_prepare_is_non_mutating_for_selected_and_ordinary_decode_rows():
    scheduler = Scheduler(_config())
    selected = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=4,
    )
    ordinary = _running_sequence(
        scheduler,
        [3, 4],
        max_tokens=4,
    )
    before = _snapshot(scheduler, (selected, ordinary))

    prepared = scheduler.prepare_postprocess(
        (selected, ordinary),
        (
            ScheduledOutputRow(
                sequence_id=selected.seq_id,
                output_tokens=(11, 12, 13),
                speculative=True,
                accepted_draft_tokens=(11, 12),
            ),
            ScheduledOutputRow(
                sequence_id=ordinary.seq_id,
                output_tokens=(21,),
                speculative=False,
            ),
        ),
    )

    assert isinstance(prepared, PreparedSchedulerPostprocess)
    assert prepared.state == "prepared"
    assert prepared.scheduled_sequence_ids == (
        selected.seq_id,
        ordinary.seq_id,
    )
    assert _snapshot(scheduler, (selected, ordinary)) == before


def test_prepare_postprocess_does_not_iterate_all_blocks():
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "num_kvcache_blocks": 4096,
            }
        )
    )
    sequence = _running_sequence(scheduler, [1, 2])
    guarded = _IndexableNonIterableBlocks(
        scheduler.block_manager.blocks
    )
    scheduler.block_manager.blocks = guarded

    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11,),
                speculative=False,
            ),
        ),
    )

    assert isinstance(
        prepared.snapshot,
        scheduler_module.SchedulerPostprocessJournal,
    )
    assert set(guarded.index_reads) <= set(sequence.block_table)
    assert prepared.snapshot.touched_block_count == len(
        sequence.block_table
    )


def test_prepare_postprocess_does_not_scan_free_blocks():
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "num_kvcache_blocks": 4096,
            }
        )
    )
    sequence = _running_sequence(scheduler, [1, 2])
    scheduler.block_manager.free_block_ids = (
        _NonSearchableFreeBlocks(
            scheduler.block_manager.free_block_ids
        )
    )

    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11,),
                speculative=False,
            ),
        ),
    )

    assert prepared.snapshot.touched_block_count == len(
        sequence.block_table
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        (
            lambda first, second: (
                ScheduledOutputRow(
                    sequence_id=second.seq_id,
                    output_tokens=(21,),
                    speculative=False,
                ),
                ScheduledOutputRow(
                    sequence_id=first.seq_id,
                    output_tokens=(11,),
                    speculative=True,
                ),
            ),
            "exactly match scheduled sequence order",
        ),
        (
            lambda first, second: (
                ScheduledOutputRow(
                    sequence_id=first.seq_id,
                    output_tokens=(11,),
                    speculative=True,
                ),
            ),
            "exactly match scheduled sequence order",
        ),
        (
            lambda first, second: (
                ScheduledOutputRow(
                    sequence_id=first.seq_id,
                    output_tokens=(11,),
                    speculative=True,
                ),
                ScheduledOutputRow(
                    sequence_id=second.seq_id,
                    output_tokens=(21, 22),
                    speculative=False,
                ),
            ),
            "ordinary output must contain exactly one token",
        ),
    ),
)
def test_prepare_rejects_invalid_row_partition(rows, message):
    scheduler = Scheduler(_config())
    first = _running_sequence(scheduler, [1, 2])
    second = _running_sequence(scheduler, [3, 4])

    with pytest.raises(ValueError, match=message):
        scheduler.prepare_postprocess(
            (first, second),
            rows(first, second),
        )


def test_prepare_rejects_selected_non_greedy_row():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        temperature=0.6,
    )

    with pytest.raises(ValueError, match="greedy temperature"):
        scheduler.prepare_postprocess(
            (sequence,),
            (
                ScheduledOutputRow(
                    sequence_id=sequence.seq_id,
                    output_tokens=(11,),
                    speculative=True,
                ),
            ),
        )


def test_prepare_rejects_selected_prefill_or_non_sampling_row():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(scheduler, [1, 2])
    row = ScheduledOutputRow(
        sequence_id=sequence.seq_id,
        output_tokens=(11,),
        speculative=True,
    )

    with pytest.raises(ValueError, match="decode sampling"):
        scheduler.prepare_postprocess(
            (sequence,),
            (row,),
            is_prefill=True,
            do_sample=True,
        )
    with pytest.raises(ValueError, match="decode sampling"):
        scheduler.prepare_postprocess(
            (sequence,),
            (row,),
            is_prefill=False,
            do_sample=False,
        )


def test_prepare_rejects_budget_overflow_and_tokens_after_eos():
    scheduler = Scheduler(_config())
    budget_limited = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=2,
    )
    eos_limited = _running_sequence(
        scheduler,
        [3, 4],
        max_tokens=4,
    )

    with pytest.raises(ValueError, match="remaining output budget"):
        scheduler.prepare_postprocess(
            (budget_limited,),
            (
                ScheduledOutputRow(
                    sequence_id=budget_limited.seq_id,
                    output_tokens=(11, 12, 13),
                    speculative=True,
                ),
            ),
        )
    with pytest.raises(ValueError, match="after effective EOS"):
        scheduler.prepare_postprocess(
            (eos_limited,),
            (
                ScheduledOutputRow(
                    sequence_id=eos_limited.seq_id,
                    output_tokens=(99, 100),
                    speculative=True,
                ),
            ),
        )


def test_commit_appends_selected_tokens_once_and_requeues_rows():
    scheduler = Scheduler(_config())
    selected = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=5,
    )
    ordinary = _running_sequence(
        scheduler,
        [3, 4],
        max_tokens=5,
    )
    prepared = scheduler.prepare_postprocess(
        (selected, ordinary),
        (
            ScheduledOutputRow(
                sequence_id=selected.seq_id,
                output_tokens=(11, 12, 13),
                speculative=True,
                accepted_draft_tokens=(11, 12),
            ),
            ScheduledOutputRow(
                sequence_id=ordinary.seq_id,
                output_tokens=(21,),
                speculative=False,
            ),
        ),
    )

    scheduler.commit_prepared_postprocess(prepared)

    assert selected.completion_token_ids == [11, 12, 13]
    assert ordinary.completion_token_ids == [21]
    assert selected.status == SequenceStatus.RUNNING
    assert ordinary.status == SequenceStatus.RUNNING
    assert tuple(scheduler.running) == (selected, ordinary)
    assert prepared.state == "committed"
    with pytest.raises(RuntimeError, match="not active"):
        scheduler.commit_prepared_postprocess(prepared)


def test_commit_finishes_on_eos_and_releases_storage_once():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=5,
    )
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 99),
                speculative=True,
                accepted_draft_tokens=(11, 99),
            ),
        ),
    )

    scheduler.commit_prepared_postprocess(prepared)

    assert sequence.completion_token_ids == [11, 99]
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert tuple(scheduler.running) == ()
    assert prepared.state == "committed"


def test_rollback_prepared_postprocess_is_non_mutating():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(scheduler, [1, 2])
    before = _snapshot(scheduler, (sequence,))
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 12),
                speculative=True,
                accepted_draft_tokens=(11,),
            ),
        ),
    )

    scheduler.rollback_prepared_postprocess(prepared)

    assert prepared.state == "rolled_back"
    assert _snapshot(scheduler, (sequence,)) == before
    with pytest.raises(RuntimeError, match="not active"):
        scheduler.rollback_prepared_postprocess(prepared)


def test_commit_failure_restores_all_prior_row_mutations(monkeypatch):
    scheduler = Scheduler(_config())
    first = _running_sequence(scheduler, [1, 2])
    second = _running_sequence(scheduler, [3, 4])
    before = _snapshot(scheduler, (first, second))
    prepared = scheduler.prepare_postprocess(
        (first, second),
        (
            ScheduledOutputRow(
                sequence_id=first.seq_id,
                output_tokens=(11, 12),
                speculative=True,
                accepted_draft_tokens=(11,),
            ),
            ScheduledOutputRow(
                sequence_id=second.seq_id,
                output_tokens=(21,),
                speculative=False,
            ),
        ),
    )
    original_append = Sequence.append_token

    def fail_second(sequence, token_id):
        if sequence is second:
            raise RuntimeError("injected append failure")
        return original_append(sequence, token_id)

    monkeypatch.setattr(Sequence, "append_token", fail_second)

    with pytest.raises(
        RuntimeError,
        match="injected append failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "commit_failed"
    assert _snapshot(scheduler, (first, second)) == before


def test_commit_failure_restores_hybrid_release(monkeypatch):
    allocator = _HybridStateSlotAllocator(capacity=4)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=4,
    )
    lease = allocator.allocate(sequence.seq_id)
    sequence.hybrid_state_slot_id = lease.slot_id
    sequence.hybrid_state_generation = lease.generation
    before = _snapshot(scheduler, (sequence,))
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(99,),
                speculative=False,
            ),
        ),
    )

    def fail_after_release(*_args, **_kwargs):
        raise RuntimeError("injected post-release failure")

    monkeypatch.setattr(
        scheduler,
        "_remove_finished_progress",
        fail_after_release,
    )

    with pytest.raises(
        RuntimeError,
        match="injected post-release failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "commit_failed"
    assert _snapshot(scheduler, (sequence,)) == before
    assert allocator.lease_for_request(sequence.seq_id) == lease


def test_commit_failure_restores_multiple_hybrid_releases(
    monkeypatch,
):
    allocator = _HybridStateSlotAllocator(capacity=4)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    first = _running_sequence(scheduler, [1, 2])
    second = _running_sequence(scheduler, [3, 4])
    first_lease = allocator.allocate(first.seq_id)
    second_lease = allocator.allocate(second.seq_id)
    for sequence, lease in (
        (first, first_lease),
        (second, second_lease),
    ):
        sequence.hybrid_state_slot_id = lease.slot_id
        sequence.hybrid_state_generation = lease.generation
    before = _snapshot(scheduler, (first, second))
    prepared = scheduler.prepare_postprocess(
        (first, second),
        (
            ScheduledOutputRow(
                sequence_id=first.seq_id,
                output_tokens=(99,),
                speculative=False,
            ),
            ScheduledOutputRow(
                sequence_id=second.seq_id,
                output_tokens=(99,),
                speculative=False,
            ),
        ),
    )
    original_remove = scheduler._remove_finished_progress

    def fail_after_second_release(
        sequence,
        removed_entries,
    ):
        if sequence is second:
            raise RuntimeError(
                "injected second hybrid release failure"
            )
        return original_remove(sequence, removed_entries)

    monkeypatch.setattr(
        scheduler,
        "_remove_finished_progress",
        fail_after_second_release,
    )

    with pytest.raises(
        RuntimeError,
        match="injected second hybrid release failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "commit_failed"
    assert _snapshot(scheduler, (first, second)) == before
    assert allocator.lease_for_request(first.seq_id) == first_lease
    assert allocator.lease_for_request(second.seq_id) == second_lease


def test_prefill_hook_failure_restores_scheduler_state(
    monkeypatch,
):
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "max_num_prefill_tokens_per_step": 2,
            }
        )
    )
    sequence = _scheduled_prefill_sequence(
        scheduler,
        [1, 2],
        chunk_end=2,
        final=True,
        do_sample=False,
    )
    before = _snapshot(scheduler, (sequence,))

    def fail_hook(_sequence):
        raise RuntimeError("prefill hook failed")

    scheduler.install_prefill_commit_hook(fail_hook)
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(),
                speculative=False,
            ),
        ),
        is_prefill=True,
        do_sample=False,
    )

    with pytest.raises(
        RuntimeError,
        match="prefill hook failed",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "commit_failed"
    expected = dict(before)
    expected["prefill_hook_error"] = (
        "RuntimeError: prefill hook failed"
    )
    assert _snapshot(scheduler, (sequence,)) == expected


def test_scheduler_journal_rollback_failure_is_terminal(
    monkeypatch,
):
    scheduler = Scheduler(_config())
    sequence = _running_sequence(scheduler, [1, 2])
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11,),
                speculative=False,
            ),
        ),
    )
    original_rollback = prepared.snapshot.rollback

    def fail_rollback(owner):
        prepared.snapshot.state = "rollback_failed"
        raise RuntimeError("injected journal rollback failure")

    monkeypatch.setattr(
        prepared.snapshot,
        "rollback",
        fail_rollback,
    )

    def fail_append(_sequence, _token_id):
        raise RuntimeError("injected scheduler failure")

    monkeypatch.setattr(Sequence, "append_token", fail_append)

    with pytest.raises(
        scheduler_module.SchedulerPostprocessRollbackError,
        match="injected journal rollback failure",
    ) as exc_info:
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "rollback_failed"
    assert str(exc_info.value.commit_error) == (
        "injected scheduler failure"
    )
    assert str(exc_info.value.rollback_error) == (
        "injected journal rollback failure"
    )
    monkeypatch.setattr(
        prepared.snapshot,
        "rollback",
        original_rollback,
    )
    with pytest.raises(RuntimeError, match="not active"):
        scheduler.rollback_prepared_postprocess(prepared)


def test_mixed_commit_handles_prefill_and_multi_token_decode():
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "max_num_prefill_tokens_per_step": 2,
                "chunked_prefill_mixed_batch": True,
            }
        )
    )
    prefill = _scheduled_prefill_sequence(
        scheduler,
        [1, 2, 3, 4],
        chunk_end=2,
        final=False,
        do_sample=False,
    )
    selected = _running_sequence(
        scheduler,
        [7, 8],
        max_tokens=5,
    )
    scheduler.running.remove(selected)
    selected.step_is_decode = True
    selected.step_do_sample = True
    prepared = scheduler.prepare_postprocess(
        (prefill, selected),
        (
            ScheduledOutputRow(
                sequence_id=prefill.seq_id,
                output_tokens=(),
                speculative=False,
            ),
            ScheduledOutputRow(
                sequence_id=selected.seq_id,
                output_tokens=(11, 12, 13),
                speculative=True,
                accepted_draft_tokens=(11, 12),
            ),
        ),
        is_prefill=True,
        do_sample=True,
        batch_kind="mixed",
    )

    scheduler.commit_prepared_postprocess(prepared)

    assert prefill.completion_token_ids == []
    assert prefill.num_computed_tokens == 2
    assert prefill.status == SequenceStatus.PREFILLING
    assert tuple(scheduler.prefilling) == (prefill,)
    assert selected.completion_token_ids == [11, 12, 13]
    assert selected.status == SequenceStatus.RUNNING
    assert tuple(scheduler.running) == (selected,)
    assert selected.step_is_decode is False
    assert selected.step_do_sample is True


def test_commit_failure_restores_state_before_external_kv_change(
    monkeypatch,
):
    scheduler = Scheduler(_config())
    sequence = _running_sequence(scheduler, [1, 2])
    extra_block_id = scheduler.block_manager.free_block_ids[0]
    scheduler.block_manager._allocate_block(extra_block_id)
    before = _snapshot(scheduler, (sequence,))
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11,),
                speculative=True,
            ),
        ),
    )
    prepared.snapshot.extend_speculative_kv_plans(
        scheduler,
        (
            SimpleNamespace(
                sequence_id=sequence.seq_id,
                committed_block_ids=(extra_block_id,),
                unused_block_ids=(),
                publications=(),
            ),
        ),
    )
    sequence.block_table.append(extra_block_id)

    def fail_append(_sequence, _token_id):
        raise RuntimeError("injected post-KV failure")

    monkeypatch.setattr(Sequence, "append_token", fail_append)

    with pytest.raises(
        RuntimeError,
        match="injected post-KV failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert _snapshot(scheduler, (sequence,)) == before


def test_legacy_postprocess_preserves_single_token_decode_behavior():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=3,
    )

    scheduler.postprocess(
        [sequence],
        [11],
        is_prefill=False,
        do_sample=True,
    )

    assert sequence.completion_token_ids == [11]
    assert sequence.status == SequenceStatus.RUNNING
    assert tuple(scheduler.running) == (sequence,)


def test_legacy_postprocess_routes_through_prepared_api(monkeypatch):
    scheduler = Scheduler(_config())
    sequence = _running_sequence(scheduler, [1, 2])
    calls = []
    prepared_marker = object()

    def prepare(
        seqs,
        rows,
        is_prefill,
        do_sample,
        batch_kind,
        *,
        decision_now_ns,
        step_end_ns,
    ):
        calls.append((
            "prepare",
            seqs,
            rows,
            is_prefill,
            do_sample,
            batch_kind,
            decision_now_ns,
            step_end_ns,
        ))
        return prepared_marker

    def commit(prepared):
        calls.append(("commit", prepared))

    monkeypatch.setattr(
        scheduler,
        "prepare_postprocess",
        prepare,
    )
    monkeypatch.setattr(
        scheduler,
        "commit_prepared_postprocess",
        commit,
    )

    scheduler.postprocess(
        [sequence],
        [11],
        is_prefill=False,
        do_sample=True,
        decision_now_ns=10,
        step_end_ns=20,
    )

    assert len(calls) == 2
    assert calls[0][0] == "prepare"
    assert calls[0][1] == (sequence,)
    assert calls[0][2] == (
        ScheduledOutputRow(
            sequence_id=sequence.seq_id,
            output_tokens=(11,),
            speculative=False,
        ),
    )
    assert calls[0][3:] == (
        False,
        True,
        None,
        10,
        20,
    )
    assert calls[1] == ("commit", prepared_marker)
    assert callable(
        getattr(
            scheduler_module.Scheduler,
            "commit_prepared_postprocess",
            None,
        )
    )
    assert callable(
        getattr(
            scheduler_module.Scheduler,
            "rollback_prepared_postprocess",
            None,
        )
    )
