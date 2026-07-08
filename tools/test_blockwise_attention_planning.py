"""Blockwise attention planning regression tests.

Run on an environment with torch/CUDA available:
  PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from tinyvllm.layers.attention import (
    _blockwise_online_decode_attention,
    _normalize_logical_block_rows,
    _stage_blockwise_read_window,
)


class _PlanOnlyManager:
    def __init__(self):
        self.gpu_blocks = 4
        self.logical_to_slot = {0: 0, 1: 1, 2: 2}
        self.stats = {
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
        }
        self.ensure_calls = []
        self.wait_calls = []

    def mark_dirty(self, blocks):
        pass

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ):
        self.ensure_calls.append(list(logical_blocks))
        return {int(block): self.logical_to_slot[int(block)] for block in logical_blocks}

    def wait_for_blocks(self, logical_blocks, clear_pending=False):
        self.wait_calls.append((list(logical_blocks), bool(clear_pending)))


def test_blockwise_decode_stages_read_window_in_first_seen_order():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [2, 0],
            [1, 2],
        ],
        kv_offload_context_lens=[2, 2],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
    )
    q = torch.ones(2, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(3, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(3, 1, 1, 1, dtype=torch.float32)

    _blockwise_online_decode_attention(
        q,
        k_cache,
        v_cache,
        context,
        num_heads=1,
        head_dim=1,
        scale=1.0,
    )

    assert manager.ensure_calls == [[2, 0, 1]]
    assert manager.wait_calls == [([2, 0, 1], True)]


def test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks():
    manager = _PlanOnlyManager()

    unique_blocks = _stage_blockwise_read_window(
        manager,
        logical_blocks=[2, 0, 2, 1],
        future_logical_blocks={0, 1, 2, 3},
        protected_logical_blocks={3},
        capacity_blocks={0, 1, 2, 3},
        capacity_error_prefix="test read window",
    )

    assert unique_blocks == [2, 0, 1]
    assert manager.stats["prefetch_plans"] == 1
    assert manager.stats["prefetch_read_blocks"] == 3
    assert manager.ensure_calls == [[2, 0, 1]]
    assert manager.wait_calls == [([2, 0, 1], True)]


def test_normalize_logical_block_rows_filters_once_and_reports_max_blocks():
    rows, max_blocks = _normalize_logical_block_rows([
        [2, -1, "3"],
        [],
        [4, 5, -1],
    ])

    assert rows == [[2, 3], [], [4, 5]]
    assert max_blocks == 2


def main():
    test_blockwise_decode_stages_read_window_in_first_seen_order()
    test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks()
    test_normalize_logical_block_rows_filters_once_and_reports_max_blocks()
    print("blockwise attention planning tests passed")


if __name__ == "__main__":
    main()
