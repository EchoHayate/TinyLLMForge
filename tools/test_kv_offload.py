"""KV offload manager regression tests.

Run on an environment with torch/CUDA available:
  PYTHONPATH=$PWD python3 tools/test_kv_offload.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from tinyvllm.engine.model_runner import KVOffloadMVP0


def _assert_raises(
    error_type,
    message: str,
    callback,
):
    try:
        callback()
    except error_type as error:
        assert message in str(error)
        return
    raise AssertionError(
        f"expected {error_type.__name__}: {message}"
    )


class _NoopKVOffload(KVOffloadMVP0):
    def __init__(self):
        self.gpu_blocks = 2
        self.logical_blocks = 4
        self.block_nbytes = 1
        self.async_copy = False
        self.batch_copy = False
        self.writeback_on_evict = False
        self.evict_policy = "lru"
        self.logical_to_slot = {}
        self.slot_to_logical = [None, None]
        self.slot_last_used = [0, 0]
        self.clock = 0
        self.cpu_valid = [False] * 4
        self.dirty_logical_blocks = set()
        self.pending_wait_blocks = set()
        self.stats = {}
        self.enqueue_d2h_calls = 0
        self.enqueue_h2d_calls = 0
        self.wait_for_blocks_calls = 0

    def _check_logical_block(self, logical_block: int):
        if logical_block < 0 or logical_block >= self.logical_blocks:
            raise IndexError(logical_block)

    def _enqueue_d2h_pairs(self, pairs):
        self.enqueue_d2h_calls += 1

    def _enqueue_h2d_pairs(self, pairs):
        self.enqueue_h2d_calls += 1

    def wait_for_blocks(self, logical_blocks, clear_pending: bool = False):
        self.wait_for_blocks_calls += 1


def _identity_manager():
    manager = _NoopKVOffload()
    manager.bound_generations = [None] * manager.logical_blocks
    manager.h2d_done = {}
    manager.d2h_done = {}
    manager.stats.update({
        "speculative_residency_prepares": 0,
        "speculative_residency_precommits": 0,
        "speculative_residency_seals": 0,
        "speculative_residency_rollbacks": 0,
        "speculative_residency_committed_blocks": 0,
        "speculative_residency_rejected_blocks": 0,
        "speculative_residency_rejected_d2h_copies": 0,
    })
    return manager


class _RecordingKVOffload(_NoopKVOffload):
    def __init__(self):
        super().__init__()
        self.gpu_blocks = 4
        self.logical_blocks = 8
        self.logical_to_slot = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
        self.slot_to_logical = [0, 1, 2, 3]
        self.slot_last_used = [3, 2, 1, 0]
        self.cpu_valid = [False, False, False, False, True, True, True, True]
        self.block_nbytes = 1
        self.batch_copy = True
        self.evict_policy = "lru"
        self.d2h_done = {}
        self.stats = {
            "evict_clean": 0,
            "evictions": 0,
            "copy_waits": 0,
        }
        self.h2d_pairs = []

    def _enqueue_h2d_pairs(self, pairs):
        self.h2d_pairs.append(list(pairs))


class _DummyStream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


def test_wait_for_blocks_clear_pending_api_without_cuda():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    stream = _DummyStream()
    manager.copy_stream = object()
    manager.h2d_done = {0: object(), 1: object()}
    manager.pending_wait_blocks = {0, 1, 2}
    manager.stats = {"copy_waits": 0}

    original_current_stream = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: stream
    try:
        manager.wait_for_blocks([0, 1], clear_pending=True)
    finally:
        torch.cuda.current_stream = original_current_stream

    assert manager.stats["copy_waits"] == 2
    assert manager.pending_wait_blocks == {2}
    assert stream.waited == [manager.h2d_done[0], manager.h2d_done[1]]


def test_wait_for_blocks_empty_is_noop_without_cuda_stream():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.copy_stream = object()
    manager.h2d_done = {}
    manager.pending_wait_blocks = {2}
    manager.stats = {"copy_waits": 0}

    original_current_stream = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: (_ for _ in ()).throw(AssertionError("current_stream called"))
    try:
        manager.wait_for_blocks([], clear_pending=True)
    finally:
        torch.cuda.current_stream = original_current_stream

    assert manager.stats["copy_waits"] == 0
    assert manager.pending_wait_blocks == {2}


def test_wait_for_blocks_without_events_clears_pending_without_cuda_stream():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.copy_stream = object()
    manager.h2d_done = {}
    manager.pending_wait_blocks = {0, 1, 2}
    manager.stats = {"copy_waits": 0}

    original_current_stream = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: (_ for _ in ()).throw(AssertionError("current_stream called"))
    try:
        manager.wait_for_blocks([0, 1], clear_pending=True)
    finally:
        torch.cuda.current_stream = original_current_stream

    assert manager.stats["copy_waits"] == 0
    assert manager.pending_wait_blocks == {2}


def test_wait_for_blocks_clear_pending_skips_non_pending_stale_events_without_cuda_stream():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.copy_stream = object()
    manager.h2d_done = {0: object(), 1: object()}
    manager.pending_wait_blocks = {1}
    manager.stats = {"copy_waits": 0}

    original_current_stream = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: (_ for _ in ()).throw(AssertionError("current_stream called"))
    try:
        manager.wait_for_blocks([0], clear_pending=True)
    finally:
        torch.cuda.current_stream = original_current_stream

    assert manager.stats["copy_waits"] == 0
    assert manager.pending_wait_blocks == {1}


def test_wait_for_blocks_clear_pending_clears_all_blocks_sharing_waited_event_without_cuda():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    stream = _DummyStream()
    event = object()
    manager.copy_stream = object()
    manager.h2d_done = {0: event, 1: event, 2: object()}
    manager.pending_wait_blocks = {0, 1, 2}
    manager.stats = {"copy_waits": 0}

    original_current_stream = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: stream
    try:
        manager.wait_for_blocks([0], clear_pending=True)
    finally:
        torch.cuda.current_stream = original_current_stream

    assert manager.stats["copy_waits"] == 1
    assert manager.pending_wait_blocks == {2}
    assert stream.waited == [event]


def test_ensure_resident_empty_blocks_is_noop_without_copy_hooks():
    manager = _NoopKVOffload()

    mapping = manager.ensure_resident([], require_valid=True, wait=True)

    assert mapping == {}
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0
    assert manager.wait_for_blocks_calls == 0


def test_ensure_resident_already_resident_blocks_skips_empty_copy_hooks():
    manager = _NoopKVOffload()
    manager.logical_to_slot = {1: 0, 2: 1}
    manager.slot_to_logical = [1, 2]

    mapping = manager.ensure_resident([1, 2, 1], require_valid=True, wait=True)

    assert mapping == {1: 0, 2: 1}
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0
    assert manager.wait_for_blocks_calls == 0


def test_summary_tracks_peak_resident_blocks_without_exceeding_capacity():
    manager = _NoopKVOffload()
    manager.stats["peak_resident_blocks"] = 0

    manager.logical_to_slot = {0: 0}
    manager._record_peak_resident_blocks()
    manager.logical_to_slot = {0: 0, 1: 1}
    manager._record_peak_resident_blocks()
    manager.logical_to_slot = {1: 1}
    manager._record_peak_resident_blocks()

    summary = manager.summary()
    assert summary["resident_blocks"] == 1
    assert summary["peak_resident_blocks"] == 2
    assert summary["gpu_blocks"] == 2


def test_peak_resident_blocks_rejects_mapping_over_capacity():
    manager = _NoopKVOffload()
    manager.stats["peak_resident_blocks"] = 0
    manager.logical_to_slot = {0: 0, 1: 1, 2: 2}

    _assert_raises(
        RuntimeError,
        "KV offload resident block count exceeds GPU capacity",
        manager._record_peak_resident_blocks,
    )


def test_ensure_resident_clean_fresh_eviction_skips_empty_copy_hooks():
    manager = _NoopKVOffload()
    manager.gpu_blocks = 1
    manager.logical_to_slot = {1: 0}
    manager.slot_to_logical = [1]
    manager.slot_last_used = [0]
    manager.evict_policy = "lru"
    manager.block_nbytes = 1
    manager.d2h_done = {}
    manager.stats = {
        "evict_clean": 0,
        "evictions": 0,
        "copy_waits": 0,
    }

    mapping = manager.ensure_resident([2], require_valid=False, wait=True)

    assert mapping == {2: 0}
    assert manager.logical_to_slot == {2: 0}
    assert manager.slot_to_logical == [2]
    assert manager.stats["evict_clean"] == 1
    assert manager.stats["evictions"] == 1
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0
    assert manager.wait_for_blocks_calls == 0


def test_bind_logical_block_identity_is_same_generation_idempotent():
    manager = _identity_manager()

    manager.bind_logical_block_identity(1, 3)
    manager.bind_logical_block_identity(1, 3)

    assert manager.bound_generations[1] == 3
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0


def test_bind_logical_block_identity_newer_generation_clears_stale_owner():
    manager = _identity_manager()
    manager.bound_generations[1] = 3
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.cpu_valid[1] = True
    manager.dirty_logical_blocks.add(1)
    manager.pending_wait_blocks.add(1)
    manager.h2d_done[1] = object()
    manager.d2h_done[1] = object()

    manager.bind_logical_block_identity(1, 4)

    assert manager.bound_generations[1] == 4
    assert 1 not in manager.logical_to_slot
    assert manager.slot_to_logical[0] is None
    assert manager.cpu_valid[1] is False
    assert 1 not in manager.dirty_logical_blocks
    assert 1 not in manager.pending_wait_blocks
    assert 1 not in manager.h2d_done
    assert 1 not in manager.d2h_done
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0


def test_bind_logical_block_identity_rejects_older_generation():
    manager = _identity_manager()
    manager.bound_generations[1] = 4

    _assert_raises(
        RuntimeError,
        "moved backwards",
        lambda: manager.bind_logical_block_identity(1, 3),
    )

    assert manager.bound_generations[1] == 4


def test_bind_logical_block_identity_rejects_unbound_existing_state():
    manager = _identity_manager()
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1

    _assert_raises(
        RuntimeError,
        "existing state",
        lambda: manager.bind_logical_block_identity(1, 1),
    )

    assert manager.bound_generations[1] is None
    assert manager.logical_to_slot == {1: 0}


def test_discard_resident_blocks_rejects_generation_mismatch_atomically():
    manager = _identity_manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1

    _assert_raises(
        RuntimeError,
        "generation mismatch",
        lambda: manager.discard_resident_blocks(
            ((1, 6),),
            allow_dirty=False,
        ),
    )

    assert manager.logical_to_slot == {1: 0}
    assert manager.slot_to_logical == [1, None]


def test_discard_resident_blocks_rejects_dirty_without_copy():
    manager = _identity_manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.dirty_logical_blocks.add(1)

    _assert_raises(
        RuntimeError,
        "dirty",
        lambda: manager.discard_resident_blocks(
            ((1, 7),),
            allow_dirty=False,
        ),
    )

    assert manager.logical_to_slot == {1: 0}
    assert manager.dirty_logical_blocks == {1}
    assert manager.enqueue_d2h_calls == 0


def test_discard_resident_blocks_clears_metadata_without_copy():
    manager = _identity_manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.pending_wait_blocks.add(1)
    manager.h2d_done[1] = object()
    manager.d2h_done[1] = object()

    discarded = manager.discard_resident_blocks(
        ((1, 7),),
        allow_dirty=False,
    )

    assert discarded == ((1, 7),)
    assert 1 not in manager.logical_to_slot
    assert manager.slot_to_logical[0] is None
    assert 1 not in manager.pending_wait_blocks
    assert 1 not in manager.h2d_done
    assert 1 not in manager.d2h_done
    assert manager.bound_generations[1] == 7
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0


def test_discard_resident_blocks_restores_snapshot_after_mutation_failure():
    manager = _identity_manager()
    for block_id, slot in ((1, 0), (2, 1)):
        manager.bound_generations[block_id] = 7
        manager.logical_to_slot[block_id] = slot
        manager.slot_to_logical[slot] = block_id
    original_discard = manager._discard_validated_resident_block
    calls = 0

    def fail_second(block_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected discard failure")
        return original_discard(block_id)

    manager._discard_validated_resident_block = fail_second

    _assert_raises(
        RuntimeError,
        "injected discard failure",
        lambda: manager.discard_resident_blocks(
            ((1, 7), (2, 7)),
            allow_dirty=False,
        ),
    )

    assert manager.logical_to_slot == {1: 0, 2: 1}
    assert manager.slot_to_logical == [1, 2]


def _clean_resident_identity_manager():
    manager = _identity_manager()
    manager.stats.update({
        "evictions": 0,
        "evict_clean": 0,
    })
    for block_id, slot in ((1, 0), (2, 1)):
        manager.bound_generations[block_id] = 7
        manager.logical_to_slot[block_id] = slot
        manager.slot_to_logical[slot] = block_id
        manager.cpu_valid[block_id] = True
        manager.d2h_done[block_id] = object()
    return manager


def test_evict_clean_resident_blocks_preserves_cpu_generation():
    manager = _clean_resident_identity_manager()

    evicted = manager.evict_clean_resident_blocks(
        ((2, 7), (1, 7)),
    )

    assert evicted == ((2, 7), (1, 7))
    assert manager.logical_to_slot == {}
    assert manager.slot_to_logical == [None, None]
    assert manager.cpu_valid[1] is True
    assert manager.cpu_valid[2] is True
    assert manager.bound_generations[1] == 7
    assert manager.bound_generations[2] == 7
    assert manager.d2h_done == {}
    assert manager.stats["evictions"] == 2
    assert manager.stats["evict_clean"] == 2


def test_evict_clean_resident_blocks_rejects_dirty_batch_atomically():
    manager = _clean_resident_identity_manager()
    manager.dirty_logical_blocks.add(2)

    _assert_raises(
        RuntimeError,
        "clean",
        lambda: manager.evict_clean_resident_blocks(
            ((1, 7), (2, 7)),
        ),
    )

    assert manager.logical_to_slot == {1: 0, 2: 1}
    assert manager.slot_to_logical == [1, 2]
    assert manager.stats["evictions"] == 0
    assert manager.stats["evict_clean"] == 0


def test_ensure_resident_assigns_contiguous_missing_blocks_to_contiguous_slots_for_coalesced_h2d():
    manager = _RecordingKVOffload()

    mapping = manager.ensure_resident([4, 5, 6, 7], require_valid=True, wait=False)

    assert mapping == {4: 0, 5: 1, 6: 2, 7: 3}
    assert manager.h2d_pairs == [[(4, 0), (5, 1), (6, 2), (7, 3)]]
    assert manager._coalesce_copy_pairs(manager.h2d_pairs[0]) == [(4, 0, 4)]


def test_future_only_block_biases_victim_score_without_becoming_protected():
    manager = _RecordingKVOffload()
    manager.evict_policy = "lru_cost"
    manager.slot_last_used = [0, 1, 2, 3]
    manager.cpu_valid[4] = True

    mapping = manager.ensure_resident(
        [4],
        require_valid=True,
        future_logical_blocks={0},
        protected_logical_blocks=set(),
        wait=False,
    )

    assert mapping == {4: 1}
    assert 0 in manager.logical_to_slot
    assert 4 in manager.logical_to_slot
    assert manager.h2d_pairs == [[(4, 1)]]
    assert manager.pending_wait_blocks == set()


def test_future_only_missing_blocks_are_not_loaded_pending_or_waited():
    manager = _RecordingKVOffload()
    manager.cpu_valid[4] = True
    manager.cpu_valid[5] = True

    mapping = manager.ensure_resident(
        [4],
        require_valid=True,
        future_logical_blocks={5},
        protected_logical_blocks=set(),
        wait=False,
    )

    assert mapping == {4: manager.logical_to_slot[4]}
    assert 5 not in manager.logical_to_slot
    assert all(
        logical != 5
        for batch in manager.h2d_pairs
        for logical, _ in batch
    )
    assert 5 not in manager.pending_wait_blocks


def test_map_block_rows_uses_existing_resident_slots_without_staging():
    manager = _NoopKVOffload()
    manager.block_size = 8
    manager.logical_to_slot = {3: 1, 5: 0}

    rows = manager.map_block_rows([[3, -1, 5], [5, 3, -1]])

    assert rows == [[1, -1, 0], [0, 1, -1]]
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0
    assert manager.wait_for_blocks_calls == 0


def test_map_slots_for_positions_uses_existing_resident_slots_without_staging():
    manager = _NoopKVOffload()
    manager.block_size = 8
    manager.logical_to_slot = {3: 1, 5: 0}

    slots = manager.map_slots_for_positions([3, 5], [0, 7, 8, 15])

    assert slots == [8, 15, 0, 7]
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0
    assert manager.wait_for_blocks_calls == 0


def test_dirty_evictions_are_batched_when_loading_multiple_blocks():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])

    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    manager.synchronize_copies()

    summary = manager.summary()
    assert summary["evict_dirty"] == 2
    assert summary["d2h_copies"] == 2
    assert summary["d2h_batches"] == 1
    assert summary["d2h_batch_spans"] == 1


def test_deferred_clean_eviction_waits_for_pending_d2h_event():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])

    before_waits = manager.summary()["copy_waits"]
    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    manager.synchronize_copies()

    summary = manager.summary()
    assert summary["evict_clean"] == 2
    assert summary["copy_waits"] == before_waits + 1


def test_wait_for_blocks_coalesces_identical_h2d_events():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])
    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    manager.ensure_resident([0, 1], require_valid=True, future_logical_blocks={0, 1}, wait=False)

    assert manager.h2d_done[0] is manager.h2d_done[1]
    before_waits = manager.summary()["copy_waits"]
    manager.wait_for_blocks([0, 1])
    after_waits = manager.summary()["copy_waits"]
    assert after_waits == before_waits + 1


def test_deferred_eviction_waits_once_per_identical_d2h_event():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])

    assert manager.d2h_done[0] is manager.d2h_done[1]
    before_waits = manager.summary()["copy_waits"]
    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    after_waits = manager.summary()["copy_waits"]
    assert after_waits == before_waits + 1


def test_ensure_resident_wait_clears_pending_h2d_waits():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])
    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    manager.ensure_resident([0, 1], require_valid=True, future_logical_blocks={0, 1}, wait=True)

    assert manager.pending_wait_blocks.isdisjoint({0, 1})
    before_waits = manager.summary()["copy_waits"]
    manager.wait_for_pending()
    after_waits = manager.summary()["copy_waits"]
    assert after_waits == before_waits


def test_wait_for_blocks_clears_pending_blocks_that_share_h2d_event():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])
    manager.ensure_resident([2, 3], require_valid=False, future_logical_blocks={2, 3}, wait=True)
    manager.ensure_resident([0, 1], require_valid=True, future_logical_blocks={0, 1}, wait=False)

    assert manager.pending_wait_blocks == {0, 1}
    assert manager.h2d_done[0] is manager.h2d_done[1]
    before_waits = manager.summary()["copy_waits"]
    manager.wait_for_blocks([0], clear_pending=True)
    after_waits = manager.summary()["copy_waits"]

    assert after_waits == before_waits + 1
    assert manager.pending_wait_blocks == set()


def test_evict_policy_avoids_pending_h2d_block_when_possible():
    if not torch.cuda.is_available():
        print("skipping CUDA-only KV offload test")
        return
    torch.cuda.set_device(0)
    kv_cache = torch.empty(2, 1, 2, 4, 1, 2, dtype=torch.float16, device="cuda")
    manager = KVOffloadMVP0(kv_cache, logical_blocks=4, block_size=4, async_copy=True, batch_copy=True)

    manager.ensure_resident([0, 1], require_valid=False, future_logical_blocks={0, 1}, wait=True)
    for logical_block in (0, 1):
        kv_cache[:, :, manager.logical_to_slot[logical_block]].fill_(float(logical_block + 1))
    manager.mark_dirty([0, 1])
    manager.writeback_dirty([0, 1])
    manager.ensure_resident([2], require_valid=False, future_logical_blocks={2}, wait=True)
    manager.ensure_resident([0], require_valid=True, future_logical_blocks={0}, wait=False)

    assert 0 in manager.pending_wait_blocks
    assert 2 in manager.logical_to_slot
    manager.slot_last_used[manager.logical_to_slot[0]] = 0
    manager.ensure_resident([3], require_valid=False, future_logical_blocks={3}, wait=True)
    assert 0 in manager.logical_to_slot
    assert 2 not in manager.logical_to_slot


def main():
    test_wait_for_blocks_clear_pending_api_without_cuda()
    test_wait_for_blocks_empty_is_noop_without_cuda_stream()
    test_wait_for_blocks_without_events_clears_pending_without_cuda_stream()
    test_wait_for_blocks_clear_pending_skips_non_pending_stale_events_without_cuda_stream()
    test_wait_for_blocks_clear_pending_clears_all_blocks_sharing_waited_event_without_cuda()
    test_ensure_resident_empty_blocks_is_noop_without_copy_hooks()
    test_ensure_resident_already_resident_blocks_skips_empty_copy_hooks()
    test_summary_tracks_peak_resident_blocks_without_exceeding_capacity()
    test_peak_resident_blocks_rejects_mapping_over_capacity()
    test_ensure_resident_clean_fresh_eviction_skips_empty_copy_hooks()
    test_bind_logical_block_identity_is_same_generation_idempotent()
    test_bind_logical_block_identity_newer_generation_clears_stale_owner()
    test_bind_logical_block_identity_rejects_older_generation()
    test_bind_logical_block_identity_rejects_unbound_existing_state()
    test_discard_resident_blocks_rejects_generation_mismatch_atomically()
    test_discard_resident_blocks_rejects_dirty_without_copy()
    test_discard_resident_blocks_clears_metadata_without_copy()
    test_discard_resident_blocks_restores_snapshot_after_mutation_failure()
    test_evict_clean_resident_blocks_preserves_cpu_generation()
    test_evict_clean_resident_blocks_rejects_dirty_batch_atomically()
    test_ensure_resident_assigns_contiguous_missing_blocks_to_contiguous_slots_for_coalesced_h2d()
    test_future_only_block_biases_victim_score_without_becoming_protected()
    test_future_only_missing_blocks_are_not_loaded_pending_or_waited()
    test_map_block_rows_uses_existing_resident_slots_without_staging()
    test_map_slots_for_positions_uses_existing_resident_slots_without_staging()
    test_dirty_evictions_are_batched_when_loading_multiple_blocks()
    test_deferred_clean_eviction_waits_for_pending_d2h_event()
    test_wait_for_blocks_coalesces_identical_h2d_events()
    test_deferred_eviction_waits_once_per_identical_d2h_event()
    test_ensure_resident_wait_clears_pending_h2d_waits()
    test_wait_for_blocks_clears_pending_blocks_that_share_h2d_event()
    test_evict_policy_avoids_pending_h2d_block_when_possible()
    print("kv offload tests passed")


if __name__ == "__main__":
    main()
