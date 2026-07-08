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
    assert summary["copy_waits"] >= before_waits + 2


def main():
    test_dirty_evictions_are_batched_when_loading_multiple_blocks()
    test_deferred_clean_eviction_waits_for_pending_d2h_event()
    print("kv offload tests passed")


if __name__ == "__main__":
    main()
