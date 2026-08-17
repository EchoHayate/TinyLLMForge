import importlib.util
import hashlib
import os
import sys
import types
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(ROOT, "tinyvllm", "engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)

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


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(ROOT, relative_path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sampling = load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_module = load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
hybrid_state = load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
scheduler_module = load_module(
    "tinyvllm.engine.scheduler",
    "tinyvllm/engine/scheduler.py",
)

SamplingParams = sampling.SamplingParams
Sequence = sequence_module.Sequence
SequenceStatus = sequence_module.SequenceStatus
Scheduler = scheduler_module.Scheduler
HybridStateLease = hybrid_state.HybridStateLease
HybridStateSlotAllocator = hybrid_state.HybridStateSlotAllocator
Sequence.block_size = 2


def make_config(**overrides):
    values = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 64,
        "max_model_len": 64,
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_adaptive_enter_waiting": 8,
        "chunked_prefill_adaptive_exit_waiting": 2,
        "chunked_prefill_adaptive_transition_steps": 2,
        "chunked_prefill_adaptive_max_mixed_steps": 2,
        "chunked_prefill_slo_mixed": False,
        "chunked_prefill_slo_target_gap_ns": 0,
        "chunked_prefill_slo_reserve_ns": 0,
        "chunked_prefill_slo_cost_intercept_ns": 0,
        "chunked_prefill_slo_cost_per_prefill_token_ns": 0,
        "chunked_prefill_slo_min_chunk_tokens": 1,
        "eos": -1,
        "num_kvcache_blocks": 8,
        "kvcache_block_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_sequence(tokens, max_tokens=4):
    return Sequence(
        list(tokens),
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def lease_of(sequence):
    return HybridStateLease(
        slot_id=sequence.hybrid_state_slot_id,
        generation=sequence.hybrid_state_generation,
        request_id=sequence.seq_id,
    )


def test_disabled_scheduler_observation_is_unchanged():
    scheduler = Scheduler(make_config())
    snapshot = scheduler.observation_snapshot()
    assert "hybrid_state" not in snapshot
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    scheduled, is_prefill, do_sample = scheduler.schedule()
    assert scheduled == [sequence]
    assert (is_prefill, do_sample) == (True, True)
    assert sequence.hybrid_state_slot_id == -1


def test_state_exhaustion_does_not_consume_candidate_kv():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(max_num_seqs=2),
        hybrid_state_allocator=allocator,
    )
    first = make_sequence([1, 2])
    second = make_sequence([3, 4])
    scheduler.add(first)
    scheduler.add(second)
    scheduled, _, _ = scheduler.schedule()
    assert scheduled == [first]
    assert list(scheduler.waiting) == [second]
    assert second.block_table == []
    assert second.hybrid_state_slot_id == -1
    assert len(scheduler.block_manager.used_block_ids) == first.num_blocks


def test_kv_allocation_failure_rolls_back_new_lease():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2])
    original_allocate = scheduler.block_manager.allocate

    def fail_allocate(*args, **kwargs):
        raise RuntimeError("injected KV allocation failure")

    scheduler.block_manager.allocate = fail_allocate
    try:
        scheduler._allocate_request_storage(
            sequence,
            publish_hashes=False,
            max_cached_tokens=0,
        )
    except RuntimeError as error:
        assert str(error) == "injected KV allocation failure"
    else:
        raise AssertionError("KV allocation failure was swallowed")
    finally:
        scheduler.block_manager.allocate = original_allocate
    assert sequence.hybrid_state_slot_id == -1
    assert allocator.can_allocate()
    assert allocator.lease_for_request(sequence.seq_id) is None


def test_chunked_prefill_preserves_lease_and_finish_releases_both():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=2),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2, 3, 4], max_tokens=1)
    scheduler.add(sequence)
    scheduled, is_prefill, do_sample = scheduler.schedule()
    assert scheduled == [sequence]
    assert (is_prefill, do_sample) == (True, False)
    first_lease = lease_of(sequence)
    scheduler.postprocess(
        scheduled,
        token_ids=None,
        is_prefill=True,
        do_sample=False,
    )
    scheduled, is_prefill, do_sample = scheduler.schedule()
    assert lease_of(sequence) == first_lease
    assert (is_prefill, do_sample) == (True, True)
    scheduler.postprocess(
        scheduled,
        token_ids=[9],
        is_prefill=True,
        do_sample=True,
    )
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert sequence.hybrid_state_slot_id == -1
    assert allocator.can_allocate()


def test_preemption_releases_and_readmission_increments_generation():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    scheduler.schedule()
    first = lease_of(sequence)
    scheduler.running.remove(sequence)
    scheduler.preempt(sequence)
    assert sequence.block_table == []
    assert sequence.hybrid_state_slot_id == -1
    scheduler.schedule()
    second = lease_of(sequence)
    assert second.slot_id == first.slot_id
    assert second.generation == first.generation + 1


def test_invalid_release_lease_preserves_kv_and_allocator_state():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    scheduler.schedule()
    original_blocks = list(sequence.block_table)
    original_lease = allocator.lease_for_request(sequence.seq_id)
    sequence.hybrid_state_generation += 1
    try:
        scheduler._release_request_storage(sequence)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid hybrid release metadata was accepted")
    assert sequence.block_table == original_blocks
    assert allocator.lease_for_request(sequence.seq_id) == original_lease
    assert set(original_blocks).issubset(
        scheduler.block_manager.used_block_ids
    )


def test_hybrid_prefix_reuse_fails_closed():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    seed = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.commit_prefill(seed, 0, len(seed))
    scheduler.block_manager.deallocate(seed)

    candidate = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.add(candidate)
    try:
        scheduler.schedule()
    except RuntimeError as error:
        assert "hybrid prefix reuse requires aligned state snapshot" in str(error)
    else:
        raise AssertionError("hybrid prefix reuse was accepted")
    assert candidate.block_table == []
    assert candidate.hybrid_state_slot_id == -1
    assert allocator.can_allocate()


def test_chunked_hybrid_prefix_failure_preserves_waiting_request():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=2),
        hybrid_state_allocator=allocator,
    )
    seed = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.commit_prefill(seed, 0, len(seed))
    scheduler.block_manager.deallocate(seed)

    candidate = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.add(candidate)
    try:
        scheduler.schedule()
    except RuntimeError as error:
        assert "hybrid prefix reuse requires aligned state snapshot" in str(error)
    else:
        raise AssertionError("chunked hybrid prefix reuse was accepted")
    assert list(scheduler.waiting) == [candidate]
    assert candidate.block_table == []
    assert candidate.hybrid_state_slot_id == -1
    assert allocator.can_allocate()


def test_restored_prefix_resources_are_admitted_without_reallocation():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.block_manager.allocate(
        sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    sequence.num_cached_tokens = 2
    sequence.num_computed_tokens = 2
    lease = allocator.allocate(sequence.seq_id)
    sequence.hybrid_state_slot_id = lease.slot_id
    sequence.hybrid_state_generation = lease.generation
    sequence.hybrid_prefix_restore_attempted = True
    sequence.hybrid_prefix_restore_hit = True
    original_blocks = list(sequence.block_table)
    scheduler.add(sequence)

    scheduled, is_prefill, do_sample = scheduler.schedule()

    assert scheduled == [sequence]
    assert (is_prefill, do_sample) == (True, True)
    assert sequence.prefill_chunk_start == 2
    assert sequence.block_table == original_blocks
    assert allocator.lease_for_request(sequence.seq_id) == lease


def test_restore_miss_disables_kv_only_reuse_and_prefills_from_zero():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    seed = make_sequence([1, 2, 3], max_tokens=1)
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.commit_prefill(seed, 0, len(seed))
    scheduler.block_manager.deallocate(seed)
    sequence = make_sequence([1, 2, 3], max_tokens=1)
    sequence.hybrid_prefix_restore_attempted = True
    sequence.hybrid_prefix_restore_hit = False
    scheduler.add(sequence)

    scheduled, is_prefill, do_sample = scheduler.schedule()

    assert scheduled == [sequence]
    assert (is_prefill, do_sample) == (True, True)
    assert sequence.num_cached_tokens == 0
    assert sequence.prefill_chunk_start == 0


def test_observation_reports_allocator_state_only_when_enabled():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    snapshot = scheduler.observation_snapshot()
    assert snapshot["hybrid_state"]["capacity"] == 2
    assert snapshot["hybrid_state"]["free_slots"] == 2


def test_finish_publishes_exact_release_event_once():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=2),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2], max_tokens=1)
    scheduler.add(sequence)
    scheduled, is_prefill, do_sample = scheduler.schedule()
    released = lease_of(sequence)
    scheduler.postprocess(
        scheduled,
        token_ids=[9],
        is_prefill=is_prefill,
        do_sample=do_sample,
    )
    assert scheduler.drain_hybrid_state_release_events() == (released,)
    assert scheduler.drain_hybrid_state_release_events() == ()


def test_preemption_publishes_exact_release_event():
    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    scheduler.schedule()
    released = lease_of(sequence)
    scheduler.running.remove(sequence)
    scheduler.preempt(sequence)
    assert scheduler.drain_hybrid_state_release_events() == (released,)


def test_disabled_and_failed_release_publish_no_event():
    disabled = Scheduler(make_config())
    assert disabled.drain_hybrid_state_release_events() == ()

    allocator = HybridStateSlotAllocator(1)
    scheduler = Scheduler(
        make_config(),
        hybrid_state_allocator=allocator,
    )
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    scheduler.schedule()
    sequence.hybrid_state_generation += 1
    try:
        scheduler._release_request_storage(sequence)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid release unexpectedly succeeded")
    assert scheduler.drain_hybrid_state_release_events() == ()


def test_release_event_restore_prepends_preserving_fifo():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        make_config(max_num_seqs=2),
        hybrid_state_allocator=allocator,
    )
    first = make_sequence([1, 2])
    second = make_sequence([3, 4])
    scheduler.add(first)
    scheduler.add(second)
    scheduler.schedule()
    first_lease = lease_of(first)
    second_lease = lease_of(second)
    scheduler.running.remove(first)
    scheduler.preempt(first)
    drained = scheduler.drain_hybrid_state_release_events()
    assert drained == (first_lease,)
    scheduler.running.remove(second)
    scheduler.preempt(second)
    scheduler.restore_hybrid_state_release_events(drained)
    assert scheduler.drain_hybrid_state_release_events() == (
        first_lease,
        second_lease,
    )


if __name__ == "__main__":
    test_disabled_scheduler_observation_is_unchanged()
    test_state_exhaustion_does_not_consume_candidate_kv()
    test_kv_allocation_failure_rolls_back_new_lease()
    test_chunked_prefill_preserves_lease_and_finish_releases_both()
    test_preemption_releases_and_readmission_increments_generation()
    test_invalid_release_lease_preserves_kv_and_allocator_state()
    test_hybrid_prefix_reuse_fails_closed()
    test_chunked_hybrid_prefix_failure_preserves_waiting_request()
    test_observation_reports_allocator_state_only_when_enabled()
    test_finish_publishes_exact_release_event_once()
    test_preemption_publishes_exact_release_event()
    test_disabled_and_failed_release_publish_no_event()
    test_release_event_restore_prepends_preserving_fifo()
    print("hybrid state scheduler tests passed")
