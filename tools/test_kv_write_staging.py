"""KV offload write staging planning tests.

Run:
  PYTHONPATH=$PWD python3 tools/test_kv_write_staging.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tinyvllm.engine.model_runner import (
    _stage_kv_offload_write_blocks,
    _stage_kv_offload_write_positions,
)


class _FakeKVOffload:
    def __init__(self):
        self.stats = {
            "prefetch_plans": 0,
            "prefetch_write_blocks": 0,
        }
        self.block_size = 8
        self.logical_to_slot = {5: 2, 6: 3}
        self.calls = []
        self.map_calls = []

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ):
        self.calls.append((
            list(logical_blocks),
            bool(require_valid),
            set(future_logical_blocks or set()),
            set(protected_logical_blocks or set()),
        ))

    def map_slots_for_positions(self, block_table, positions):
        self.map_calls.append((list(block_table), list(positions)))
        return [
            self.logical_to_slot[int(block_table[pos // self.block_size])] * self.block_size + (pos % self.block_size)
            for pos in positions
        ]


def test_stage_kv_offload_write_blocks_splits_valid_and_fresh_without_empty_calls():
    manager = _FakeKVOffload()

    _stage_kv_offload_write_blocks(
        manager,
        write_blocks=[5, 6, 5],
        first_write_offset_by_block={5: 2, 6: 0},
        future_blocks={5, 6, 7},
    )

    assert manager.stats["prefetch_plans"] == 1
    assert manager.stats["prefetch_write_blocks"] == 2
    assert manager.calls == [
        ([5], True, {5, 6, 7}, {5, 6}),
        ([6], False, {5, 6, 7}, {5, 6}),
    ]


def test_stage_kv_offload_write_blocks_skips_empty_valid_side():
    manager = _FakeKVOffload()

    _stage_kv_offload_write_blocks(
        manager,
        write_blocks=[6],
        first_write_offset_by_block={6: 0},
        future_blocks={6},
    )

    assert manager.stats["prefetch_plans"] == 1
    assert manager.stats["prefetch_write_blocks"] == 1
    assert manager.calls == [
        ([6], False, {6}, {6}),
    ]


def test_stage_kv_offload_write_positions_stages_once_then_maps_slots():
    manager = _FakeKVOffload()

    slots, write_blocks = _stage_kv_offload_write_positions(
        manager,
        block_table=[5, 6],
        positions=[2, 8, 9, 10],
        block_size=8,
        future_blocks={5, 6, 7},
    )

    assert write_blocks == [5, 6]
    assert slots == [18, 24, 25, 26]
    assert manager.stats["prefetch_plans"] == 1
    assert manager.stats["prefetch_write_blocks"] == 2
    assert manager.calls == [
        ([5], True, {5, 6, 7}, {5, 6}),
        ([6], False, {5, 6, 7}, {5, 6}),
    ]
    assert manager.map_calls == [([5, 6], [2, 8, 9, 10])]


def main():
    test_stage_kv_offload_write_blocks_splits_valid_and_fresh_without_empty_calls()
    test_stage_kv_offload_write_blocks_skips_empty_valid_side()
    test_stage_kv_offload_write_positions_stages_once_then_maps_slots()
    print("kv write staging tests passed")


if __name__ == "__main__":
    main()
