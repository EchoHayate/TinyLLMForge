from __future__ import annotations

from collections import deque
from dataclasses import replace
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
exact_burst_module = _load_module(
    "tinyvllm.engine.exact_greedy_decode_burst",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
)
split_phase_module = _load_module(
    "tinyvllm.engine.exact_greedy_decode_burst_split_phase",
    "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
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
ExactGreedyDecodeBurstResult = (
    exact_burst_module.ExactGreedyDecodeBurstResult
)
ExactBurstPhaseTransfer = (
    split_phase_module.ExactBurstPhaseTransfer
)
ExactGreedyDecodeBurstSplitResult = (
    split_phase_module.ExactGreedyDecodeBurstSplitResult
)
build_exact_burst_publication_tickets = (
    split_phase_module.build_exact_burst_publication_tickets
)
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


def _prepare_exact_burst_lease(
    scheduler,
    sequence,
    *,
    configured_width=4,
    graph_generation=7,
    allow_single_token_gate=False,
    split_phase_enabled=False,
    ragged_coalescing_enabled=False,
):
    scheduler.schedule_generation = 1
    return scheduler.prepare_exact_greedy_decode_burst(
        (sequence,),
        schedule_generation=1,
        graph_generation=graph_generation,
        enabled=True,
        configured_width=configured_width,
        is_prefill=False,
        do_sample=True,
        batch_kind=None,
        completion_only=True,
        tensor_parallel_size=1,
        rank=0,
        graph_available=True,
        incompatible_modes=(),
        allow_single_token_gate=allow_single_token_gate,
        split_phase_enabled=split_phase_enabled,
        ragged_coalescing_enabled=ragged_coalescing_enabled,
    )


def _exact_burst_result(lease, tokens):
    return ExactGreedyDecodeBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        tokens=tuple(tokens),
        replay_count=len(tokens),
        final_input_token=tokens[-1],
        final_position=(
            lease.first_write_position + len(tokens)
        ),
        final_context_length=(
            lease.initial_sequence_length + len(tokens)
        ),
        final_physical_slot=(
            lease.last_physical_slot + 1
        ),
        graph_identity_sha256="a" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )


def test_ragged_coalescing_issues_bounded_output_tail_leases(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    for remaining, expected_width in ((7, 4), (3, 3)):
        scheduler = Scheduler(
            SimpleNamespace(
                **{
                    **vars(_config()),
                    "kvcache_block_size": 16,
                }
            )
        )
        sequence = _running_sequence(
            scheduler,
            [1, 2],
            max_tokens=remaining,
            ignore_eos=True,
        )
        lease = _prepare_exact_burst_lease(
            scheduler,
            sequence,
            configured_width=8,
            split_phase_enabled=True,
            ragged_coalescing_enabled=True,
        )
        assert lease.requested_token_count == expected_width
        assert lease.authorized_token_count == expected_width

    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=1,
        ignore_eos=True,
    )
    assert _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    ) is None
    assert scheduler.exact_greedy_decode_burst_summary()[
        "fallback_counts"
    ] == {"insufficient_output_budget": 1}


def test_ragged_coalescing_issues_bounded_block_edge_lease(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        list(range(14)),
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    )
    assert lease.requested_token_count == 3
    assert lease.authorized_token_count == 3
    assert lease.last_physical_slot // 16 == (
        lease.first_physical_slot // 16
    )


class _ReadyCompletion:
    def synchronize(self):
        return None


class _TokenMailbox:
    def __init__(self, tokens):
        self._tokens = tuple(tokens)

    def tolist(self):
        return list(self._tokens)


def _exact_burst_split_result(
    lease,
    *,
    prefix_tokens=(11, 12, 13, 14),
    suffix_tokens=(15, 16, 17, 18),
):
    prefix_ticket, suffix_ticket = (
        build_exact_burst_publication_tickets(
            parent_lease_identity_sha256=lease.identity_sha256,
            first_write_position=lease.first_write_position,
            first_physical_slot=lease.first_physical_slot,
            parent_token_count=lease.authorized_token_count,
            prefix_token_count=4,
        )
    )

    def transfer(ticket, tokens):
        return ExactBurstPhaseTransfer(
            ticket=ticket,
            mailbox_generation=1,
            token_count=len(tokens),
            byte_count=len(tokens) * 8,
            completion=_ReadyCompletion(),
            mailbox=_TokenMailbox(tokens),
        )

    return ExactGreedyDecodeBurstSplitResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        graph_identity_sha256="b" * 64,
        replay_count=8,
        prefix=transfer(prefix_ticket, prefix_tokens),
        suffix=transfer(suffix_ticket, suffix_tokens),
    )


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


def test_exact_burst_row_shape_is_distinct_from_ordinary_and_speculative():
    ordinary = ScheduledOutputRow(
        sequence_id=1,
        output_tokens=(7,),
        speculative=False,
    )
    exact = ScheduledOutputRow(
        sequence_id=1,
        output_tokens=(7, 8, 9, 10),
        speculative=False,
        exact_burst=True,
    )
    speculative = ScheduledOutputRow(
        sequence_id=1,
        output_tokens=(7, 8),
        speculative=True,
        accepted_draft_tokens=(7,),
    )

    assert ordinary.exact_burst is False
    assert exact.exact_burst is True
    assert speculative.exact_burst is False


def test_gate_only_single_replay_uses_distinct_exact_row_and_commits():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=1,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=1,
        allow_single_token_gate=True,
    )
    assert lease.authorized_token_count == 1
    result = _exact_burst_result(lease, (11,))

    prepared = scheduler.prepare_exact_greedy_decode_burst_commit(
        (sequence,),
        lease,
        result,
        gate_only_single_token=True,
    )
    assert prepared.rows == (
        ScheduledOutputRow(
            sequence_id=sequence.seq_id,
            output_tokens=(11,),
            speculative=False,
            exact_burst=True,
            exact_burst_gate_only=True,
        ),
    )

    scheduler.commit_prepared_postprocess(prepared)

    assert sequence.completion_token_ids == [11]
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["commits"] == 1
    assert summary["committed_tokens"] == 1
    assert summary["pending_leases"] == 0


@pytest.mark.parametrize(
    ("row_factory", "message"),
    (
        (
            lambda sequence: ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11,),
                speculative=False,
                exact_burst=True,
            ),
            "at least two tokens",
        ),
        (
            lambda sequence: ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 12),
                speculative=True,
                exact_burst=True,
            ),
            "cannot be speculative",
        ),
        (
            lambda sequence: ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 12),
                speculative=False,
                accepted_draft_tokens=(11,),
                exact_burst=True,
            ),
            "accepted draft tokens",
        ),
        (
            lambda sequence: ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 12),
                speculative=False,
                exact_burst=True,
            ),
            "active lease",
        ),
    ),
)
def test_prepare_rejects_invalid_exact_burst_rows(
    row_factory,
    message,
):
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        ignore_eos=True,
    )

    with pytest.raises(ValueError, match=message):
        scheduler.prepare_postprocess(
            (sequence,),
            (row_factory(sequence),),
        )


@pytest.mark.parametrize(
    ("prompt", "expected_width"),
    (
        ((1,), 4),
        ((1, 2), 3),
        ((1, 2, 3), 2),
        ((1, 2, 3, 4), None),
    ),
)
def test_exact_burst_lease_clips_at_physical_block_boundary(
    monkeypatch,
    prompt,
    expected_width,
):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = SimpleNamespace(
        **{
            **vars(_config()),
            "kvcache_block_size": 4,
        }
    )
    scheduler = Scheduler(config)
    sequence = _running_sequence(
        scheduler,
        prompt,
        max_tokens=8,
        ignore_eos=True,
    )

    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=4,
    )

    if expected_width is None:
        assert lease is None
        summary = scheduler.exact_greedy_decode_burst_summary()
        assert summary["pending_leases"] == 0
        assert summary["fallback_counts"] == {
            "authorized_width_below_two": 1,
        }
        return
    assert lease.authorized_token_count == expected_width
    assert lease.first_write_position == len(prompt) - 1
    assert lease.last_write_position == (
        len(prompt) + expected_width - 2
    )
    assert lease.first_physical_slot == (
        lease.write_block_id * 4 + len(prompt) - 1
    )
    assert lease.last_physical_slot == (
        lease.first_physical_slot + expected_width - 1
    )
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 1


def test_exact_burst_lease_rejects_wrong_sequence_and_stale_generation():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )
    result = _exact_burst_result(lease, (11, 12))
    other = Sequence(
        [9],
        SamplingParams(
            temperature=0.0,
            max_tokens=8,
            ignore_eos=True,
        ),
    )

    with pytest.raises(ValueError, match="sequence ID"):
        scheduler.prepare_exact_greedy_decode_burst_commit(
            (other,),
            lease,
            result,
        )

    scheduler.schedule_generation += 1
    with pytest.raises(ValueError, match="stale"):
        scheduler.prepare_exact_greedy_decode_burst_commit(
            (sequence,),
            lease,
            result,
        )
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 1


def test_exact_burst_row_rejects_wrong_lease_token_count():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )

    with pytest.raises(ValueError, match="token count"):
        scheduler.prepare_postprocess(
            (sequence,),
            (
                ScheduledOutputRow(
                    sequence_id=sequence.seq_id,
                    output_tokens=(11, 12, 13),
                    speculative=False,
                    exact_burst=True,
                ),
            ),
        )

    scheduler.cancel_exact_greedy_decode_burst(
        lease,
        "test_cleanup",
    )


def test_exact_burst_row_cannot_commit_without_validated_result():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )
    before = _snapshot(scheduler, (sequence,))
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (
            ScheduledOutputRow(
                sequence_id=sequence.seq_id,
                output_tokens=(11, 12),
                speculative=False,
                exact_burst=True,
            ),
        ),
    )

    with pytest.raises(ValueError, match="validated result"):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "prepared"
    assert _snapshot(scheduler, (sequence,)) == before
    scheduler.cancel_exact_greedy_decode_burst(
        lease,
        "test_cleanup",
    )


def test_exact_burst_commit_rejects_stale_block_identity():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )
    result = _exact_burst_result(lease, (11, 12))
    scheduler.block_manager.blocks[
        lease.write_block_id
    ].generation += 1

    with pytest.raises(RuntimeError, match="block identity is stale"):
        scheduler.prepare_exact_greedy_decode_burst_commit(
            (sequence,),
            lease,
            result,
        )

    scheduler.fail_exact_greedy_decode_burst(
        lease,
        terminal=True,
    )
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 0


def test_exact_burst_commit_rechecks_lease_before_mutating():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )
    result = _exact_burst_result(lease, (11, 12))
    prepared = scheduler.prepare_exact_greedy_decode_burst_commit(
        (sequence,),
        lease,
        result,
    )
    before = _snapshot(scheduler, (sequence,))
    scheduler.schedule_generation += 1

    with pytest.raises(ValueError, match="stale"):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "prepared"
    assert _snapshot(scheduler, (sequence,)) == before
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 1
    scheduler.fail_exact_greedy_decode_burst(
        lease,
        terminal=True,
    )
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 0


def test_cancel_exact_burst_requires_identical_pending_lease():
    scheduler = Scheduler(_config())
    sequence = _running_sequence(
        scheduler,
        [1],
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=2,
    )
    mismatched = _prepare_mismatched_lease(lease)

    with pytest.raises(ValueError, match="pending lease"):
        scheduler.cancel_exact_greedy_decode_burst(
            mismatched,
            "pre_replay_fallback",
        )

    scheduler.cancel_exact_greedy_decode_burst(
        lease,
        "pre_replay_fallback",
    )
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["pending_leases"] == 0
    assert summary["fallback_counts"] == {
        "pre_replay_fallback": 1,
    }


def _prepare_mismatched_lease(lease):
    values = dict(vars(lease))
    values["identity_sha256"] = "b" * 64
    return type(lease)(**values)


def test_exact_burst_commit_appends_in_order_and_publishes_materialized_block(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = SimpleNamespace(
        **{
            **vars(_config()),
            "kvcache_block_size": 4,
        }
    )
    scheduler = Scheduler(config)
    sequence = _running_sequence(
        scheduler,
        [1],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=4,
    )
    result = _exact_burst_result(
        lease,
        (11, 12, 13, 14),
    )

    prepared = scheduler.prepare_exact_greedy_decode_burst_commit(
        (sequence,),
        lease,
        result,
        host_visible_gap_ns=123,
    )
    scheduler.commit_prepared_postprocess(prepared)

    assert sequence.completion_token_ids == [11, 12, 13, 14]
    first_block = scheduler.block_manager.blocks[
        sequence.block_table[0]
    ]
    assert first_block.token_ids == [1, 11, 12, 13]
    assert first_block.hash != -1
    assert scheduler.block_manager.hash_to_block_id[
        first_block.hash
    ] == first_block.block_id
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["commits"] == 1
    assert summary["committed_tokens"] == 4
    assert summary["maximum_host_visible_gap_ns"] == 123
    assert summary["pending_leases"] == 0


def test_exact_burst_completion_releases_request_storage_once(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = SimpleNamespace(
        **{
            **vars(_config()),
            "kvcache_block_size": 4,
        }
    )
    scheduler = Scheduler(config)
    sequence = _running_sequence(
        scheduler,
        [1],
        max_tokens=4,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(scheduler, sequence)
    result = _exact_burst_result(
        lease,
        (11, 12, 13, 14),
    )
    release_count = 0
    original_release = scheduler._release_request_storage

    def count_release(value):
        nonlocal release_count
        release_count += 1
        return original_release(value)

    monkeypatch.setattr(
        scheduler,
        "_release_request_storage",
        count_release,
    )
    prepared = scheduler.prepare_exact_greedy_decode_burst_commit(
        (sequence,),
        lease,
        result,
    )

    scheduler.commit_prepared_postprocess(prepared)

    assert release_count == 1
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert tuple(scheduler.running) == ()
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 0


def test_split_phase_k8_commits_prefix_then_suffix_under_one_parent_lease(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
    )
    split_result = _exact_burst_split_result(lease)

    prefix_prepared = (
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
            host_visible_gap_ns=12_000_000,
        )
    )
    scheduler.commit_prepared_postprocess(prefix_prepared)

    assert sequence.completion_token_ids == [11, 12, 13, 14]
    assert (
        scheduler._exact_greedy_decode_burst_pending_lease
        == lease
    )
    assert (
        scheduler._exact_greedy_decode_burst_split_phase
        == "prefix_committed"
    )
    prefix_summary = scheduler.exact_greedy_decode_burst_summary()
    assert prefix_summary["prefix_commits"] == 1
    assert prefix_summary["prefix_committed_tokens"] == 4
    assert prefix_summary["pending_leases"] == 1
    assert prefix_summary["commits"] == 0

    suffix_prepared = (
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="suffix",
            tokens=(15, 16, 17, 18),
            host_visible_gap_ns=7_000_000,
        )
    )
    scheduler.commit_prepared_postprocess(suffix_prepared)

    assert sequence.completion_token_ids == [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]
    assert scheduler._exact_greedy_decode_burst_pending_lease is None
    assert scheduler._exact_greedy_decode_burst_split_phase == "idle"
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["prefix_commits"] == 1
    assert summary["suffix_commits"] == 1
    assert summary["prefix_committed_tokens"] == 4
    assert summary["suffix_committed_tokens"] == 4
    assert summary["commits"] == 1
    assert summary["committed_tokens"] == 8
    assert summary["pending_leases"] == 0


def test_ragged_coalescing_finishes_k8_tail_with_k4_then_k3(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=15,
        ignore_eos=True,
    )
    parent = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    )
    assert parent.requested_token_count == 8
    split_result = _exact_burst_split_result(parent)
    scheduler.commit_prepared_postprocess(
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            parent,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )
    )
    scheduler.commit_prepared_postprocess(
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            parent,
            split_result,
            phase="suffix",
            tokens=(15, 16, 17, 18),
        )
    )

    k4 = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    )
    assert (
        k4.requested_token_count,
        k4.authorized_token_count,
    ) == (4, 4)
    scheduler.commit_prepared_postprocess(
        scheduler.prepare_exact_greedy_decode_burst_commit(
            (sequence,),
            k4,
            _exact_burst_result(k4, (21, 22, 23, 24)),
        )
    )
    assert scheduler._exact_greedy_decode_burst_split_phase == "idle"

    k3 = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    )
    assert (
        k3.requested_token_count,
        k3.authorized_token_count,
    ) == (3, 3)
    scheduler.commit_prepared_postprocess(
        scheduler.prepare_exact_greedy_decode_burst_commit(
            (sequence,),
            k3,
            _exact_burst_result(k3, (25, 26, 27)),
        )
    )

    assert sequence.completion_token_ids == [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ]
    assert sequence.status == SequenceStatus.FINISHED
    assert scheduler._exact_greedy_decode_burst_pending_lease is None
    assert scheduler._exact_greedy_decode_burst_split_phase == "idle"
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["attempts"] == 3
    assert summary["acceptances"] == 3
    assert summary["commits"] == 3
    assert summary["committed_tokens"] == 15
    assert summary["requested_width_histogram"] == {
        "3": 1,
        "4": 1,
        "8": 1,
    }
    assert summary["authorized_width_histogram"] == {
        "3": 1,
        "4": 1,
        "8": 1,
    }
    assert summary["fallback_counts"] == {}
    assert summary["prefix_commits"] == 1
    assert summary["suffix_commits"] == 1
    assert summary["final_token_d2h_calls"] == 2


def test_split_phase_rejects_out_of_order_duplicate_and_mismatched_inputs(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=12,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
    )
    split_result = _exact_burst_split_result(lease)

    with pytest.raises(ValueError, match="prefix_committed"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="suffix",
            tokens=(15, 16, 17, 18),
        )

    wrong_prefix_ticket, wrong_suffix_ticket = (
        build_exact_burst_publication_tickets(
            parent_lease_identity_sha256=lease.identity_sha256,
            first_write_position=lease.first_write_position + 1,
            first_physical_slot=lease.first_physical_slot + 1,
            parent_token_count=8,
            prefix_token_count=4,
        )
    )
    wrong_result = replace(
        split_result,
        prefix=replace(
            split_result.prefix,
            ticket=wrong_prefix_ticket,
        ),
        suffix=replace(
            split_result.suffix,
            ticket=wrong_suffix_ticket,
        ),
    )
    with pytest.raises(
        ValueError,
        match="publication ticket does not match",
    ):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            wrong_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )

    with pytest.raises(ValueError, match="phase transfer"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13),
        )

    prefix_prepared = (
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )
    )
    scheduler.commit_prepared_postprocess(prefix_prepared)

    with pytest.raises(ValueError, match="enqueued"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )

    sequence.append_token(99)
    with pytest.raises(ValueError, match="sequence length changed"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="suffix",
            tokens=(15, 16, 17, 18),
        )


def test_split_phase_rejects_schedule_and_block_generation_drift(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    config = SimpleNamespace(
        **{
            **vars(_config()),
            "kvcache_block_size": 16,
        }
    )
    scheduler = Scheduler(config)
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
    )
    split_result = _exact_burst_split_result(lease)
    scheduler.schedule_generation += 1

    with pytest.raises(ValueError, match="stale"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )

    scheduler.schedule_generation -= 1
    scheduler.block_manager.blocks[
        lease.write_block_id
    ].generation += 1
    with pytest.raises(RuntimeError, match="block identity is stale"):
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )


def test_split_phase_commit_revalidates_prepared_phase_tokens(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
    )
    split_result = _exact_burst_split_result(lease)
    prepared = (
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase="prefix",
            tokens=(11, 12, 13, 14),
        )
    )
    prepared.rows = (
        replace(
            prepared.rows[0],
            output_tokens=(91, 92, 93, 94),
        ),
    )

    with pytest.raises(ValueError, match="phase transfer"):
        scheduler.commit_prepared_postprocess(prepared)

    assert sequence.completion_token_ids == []
    assert scheduler._exact_greedy_decode_burst_pending_lease == lease
    assert scheduler._exact_greedy_decode_burst_split_phase == "enqueued"


@pytest.mark.parametrize("failure_phase", ("prefix", "suffix"))
def test_split_phase_rollback_preserves_the_last_committed_boundary(
    monkeypatch,
    failure_phase,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        [1, 2],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
    )
    split_result = _exact_burst_split_result(lease)
    if failure_phase == "suffix":
        prefix_prepared = (
            scheduler.prepare_exact_greedy_decode_burst_phase_commit(
                (sequence,),
                lease,
                split_result,
                phase="prefix",
                tokens=(11, 12, 13, 14),
            )
        )
        scheduler.commit_prepared_postprocess(prefix_prepared)
    before = _snapshot(scheduler, (sequence,))
    before_phase = scheduler._exact_greedy_decode_burst_split_phase
    prepared = (
        scheduler.prepare_exact_greedy_decode_burst_phase_commit(
            (sequence,),
            lease,
            split_result,
            phase=failure_phase,
            tokens=(
                (11, 12, 13, 14)
                if failure_phase == "prefix"
                else (15, 16, 17, 18)
            ),
        )
    )

    def fail_publication(*_args, **_kwargs):
        raise RuntimeError("injected split publication failure")

    monkeypatch.setattr(
        scheduler.block_manager,
        "publish_full_blocks",
        fail_publication,
    )
    with pytest.raises(
        RuntimeError,
        match="injected split publication failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert prepared.state == "commit_failed"
    assert _snapshot(scheduler, (sequence,)) == before
    assert scheduler._exact_greedy_decode_burst_pending_lease == lease
    assert scheduler._exact_greedy_decode_burst_split_phase == before_phase


def test_exact_burst_commit_rollback_restores_hashes_and_keeps_lease(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = SimpleNamespace(
        **{
            **vars(_config()),
            "kvcache_block_size": 4,
        }
    )
    scheduler = Scheduler(config)
    sequence = _running_sequence(
        scheduler,
        [1],
        max_tokens=8,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(scheduler, sequence)
    result = _exact_burst_result(
        lease,
        (11, 12, 13, 14),
    )
    before = _snapshot(scheduler, (sequence,))
    original_publish = (
        scheduler.block_manager.publish_full_blocks
    )

    def publish_then_fail(*args, **kwargs):
        original_publish(*args, **kwargs)
        raise RuntimeError("injected exact burst publish failure")

    monkeypatch.setattr(
        scheduler.block_manager,
        "publish_full_blocks",
        publish_then_fail,
    )
    prepared = scheduler.prepare_exact_greedy_decode_burst_commit(
        (sequence,),
        lease,
        result,
    )

    with pytest.raises(
        RuntimeError,
        match="injected exact burst publish failure",
    ):
        scheduler.commit_prepared_postprocess(prepared)

    assert _snapshot(scheduler, (sequence,)) == before
    assert scheduler.exact_greedy_decode_burst_summary()[
        "pending_leases"
    ] == 1
    scheduler.fail_exact_greedy_decode_burst(
        lease,
        terminal=True,
    )
    summary = scheduler.exact_greedy_decode_burst_summary()
    assert summary["failures"] == 1
    assert summary["pending_leases"] == 0


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
