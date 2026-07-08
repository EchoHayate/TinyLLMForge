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


def test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits():
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
    before_waits = manager.summary()["copy_waits"]
    manager.wait_for_blocks([0], clear_pending=True)
    after_waits = manager.summary()["copy_waits"]

    assert after_waits == before_waits + 1
    assert manager.pending_wait_blocks == {1}


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
    test_dirty_evictions_are_batched_when_loading_multiple_blocks()
    test_deferred_clean_eviction_waits_for_pending_d2h_event()
    test_wait_for_blocks_coalesces_identical_h2d_events()
    test_deferred_eviction_waits_once_per_identical_d2h_event()
    test_ensure_resident_wait_clears_pending_h2d_waits()
    test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits()
    test_evict_policy_avoids_pending_h2d_block_when_possible()
    print("kv offload tests passed")


if __name__ == "__main__":
    main()
