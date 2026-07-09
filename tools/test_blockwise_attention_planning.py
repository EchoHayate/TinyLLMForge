"""Blockwise attention planning regression tests.

Run on an environment with torch/CUDA available:
  PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import tinyvllm.layers.attention as attention_mod

from tinyvllm.layers.attention import (
    _blockwise_online_decode_attention,
    _blockwise_prefill_future_hint_blocks,
    _blockwise_read_window_future_hint_blocks,
    _decode_window_mask,
    _gqa_scores_decode,
    _gqa_scores_prefill,
    _gqa_weighted_values_decode,
    _gqa_weighted_values_prefill,
    _local_causal_mask,
    _merge_attention_window,
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
        self.future_calls = []
        self.protected_calls = []
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
        self.future_calls.append(set(future_logical_blocks or set()))
        self.protected_calls.append(set(protected_logical_blocks or set()))
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


def test_blockwise_decode_read_windows_hint_capacity_bounded_future_blocks():
    manager = _PlanOnlyManager()
    manager.logical_to_slot.update({3: 3, 4: 4})
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [0, 1, 2, 3, 4],
        ],
        kv_offload_context_lens=[5],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(5, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(5, 1, 1, 1, dtype=torch.float32)

    _blockwise_online_decode_attention(
        q,
        k_cache,
        v_cache,
        context,
        num_heads=1,
        head_dim=1,
        scale=1.0,
    )

    assert manager.ensure_calls == [[0], [1], [2], [3], [4]]
    assert manager.future_calls == [
        {0, 1, 2, 3},
        {1, 2, 3, 4},
        {2, 3, 4},
        {3, 4},
        {4},
    ]
    assert manager.protected_calls == [set(), set(), set(), set(), set()]
    assert manager.wait_calls == [
        ([0], True),
        ([1], True),
        ([2], True),
        ([3], True),
        ([4], True),
    ]


def test_blockwise_decode_reuses_cached_read_window_plan_across_layers():
    manager = _PlanOnlyManager()
    manager.logical_to_slot.update({3: 3, 4: 4})
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [0, 1, 2, 3, 4],
        ],
        kv_offload_context_lens=[5],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=None,
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(5, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(5, 1, 1, 1, dtype=torch.float32)

    _blockwise_online_decode_attention(
        q,
        k_cache,
        v_cache,
        context,
        num_heads=1,
        head_dim=1,
        scale=1.0,
    )

    with patch.object(
        attention_mod,
        "_blockwise_read_window_future_hint_blocks",
        side_effect=AssertionError("decode read-window plan recomputed"),
    ):
        _blockwise_online_decode_attention(
            q,
            k_cache,
            v_cache,
            context,
            num_heads=1,
            head_dim=1,
            scale=1.0,
        )


def test_blockwise_decode_gqa_does_not_materialize_repeated_kv_heads():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1]],
        kv_offload_context_lens=[2],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
    )
    q = torch.ones(1, 4, 1, dtype=torch.float32)
    k_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)
    v_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)

    with patch.object(
        attention_mod,
        "_repeat_kv_for_gqa",
        side_effect=AssertionError("repeat_kv_for_gqa called"),
    ):
        out = attention_mod._blockwise_online_decode_attention(
            q,
            k_cache,
            v_cache,
            context,
            num_heads=4,
            head_dim=1,
            scale=1.0,
        )

    assert out.shape == (1, 4, 1)


def test_gqa_grouped_helpers_match_repeated_kv_reference():
    q_decode = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    k_decode = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2) / 10.0
    weights_decode = torch.arange(12, dtype=torch.float32).view(1, 4, 3) / 7.0
    v_decode = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2) / 5.0

    repeated_k_decode = k_decode.repeat_interleave(2, dim=2)
    repeated_v_decode = v_decode.repeat_interleave(2, dim=2)
    assert torch.allclose(
        _gqa_scores_decode(q_decode, k_decode, num_heads=4, scale=0.5),
        torch.einsum("bhd,bthd->bht", q_decode, repeated_k_decode) * 0.5,
    )
    assert torch.allclose(
        _gqa_weighted_values_decode(weights_decode, v_decode, num_heads=4),
        torch.einsum("bht,bthd->bhd", weights_decode, repeated_v_decode),
    )

    q_prefill = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
    k_prefill = torch.arange(6, dtype=torch.float32).view(3, 2, 1) / 10.0
    weights_prefill = torch.arange(24, dtype=torch.float32).view(2, 4, 3) / 7.0
    v_prefill = torch.arange(6, dtype=torch.float32).view(3, 2, 1) / 5.0
    repeated_k_prefill = k_prefill.repeat_interleave(2, dim=1)
    repeated_v_prefill = v_prefill.repeat_interleave(2, dim=1)
    assert torch.allclose(
        _gqa_scores_prefill(q_prefill, k_prefill, num_heads=4, scale=0.5),
        torch.einsum("qhd,thd->qht", q_prefill, repeated_k_prefill) * 0.5,
    )
    assert torch.allclose(
        _gqa_weighted_values_prefill(weights_prefill, v_prefill, num_heads=4),
        torch.einsum("qht,thd->qhd", weights_prefill, repeated_v_prefill),
    )


def test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks():
    manager = _PlanOnlyManager()

    unique_blocks = _stage_blockwise_read_window(
        manager,
        logical_blocks=[2, 0, 2, 1],
        future_extra_blocks={3},
        protected_extra_blocks={3},
        capacity_extra_blocks={3},
        capacity_error_prefix="test read window",
    )

    assert unique_blocks == [2, 0, 1]
    assert manager.stats["prefetch_plans"] == 1
    assert manager.stats["prefetch_read_blocks"] == 3
    assert manager.ensure_calls == [[2, 0, 1]]
    assert manager.future_calls == [{0, 1, 2, 3}]
    assert manager.protected_calls == [{3}]
    assert manager.wait_calls == [([2, 0, 1], True)]


def test_blockwise_prefill_future_hint_blocks_fill_only_spare_capacity():
    future_blocks = _blockwise_read_window_future_hint_blocks(
        row_blocks=[0, 1, 2, 3, 4],
        start_block=0,
        stop_block=5,
        window_blocks=1,
        extra_future_blocks={4},
        gpu_blocks=4,
    )

    assert future_blocks == {1, 2, 4}

    assert _blockwise_prefill_future_hint_blocks(
        row_blocks=[0, 1, 2, 3, 4],
        start_block=0,
        prefix_blocks=5,
        window_blocks=1,
        write_blocks={4},
        gpu_blocks=4,
    ) == {1, 2, 4}


def test_blockwise_prefill_read_windows_hint_next_prefix_blocks():
    manager = _PlanOnlyManager()
    manager.logical_to_slot.update({3: 3})
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1, 2, 3]],
        kv_offload_prefill_chunk_starts=[4],
        kv_offload_prefill_chunk_ends=[5],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k = torch.ones(1, 1, 1, dtype=torch.float32)
    v = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.ones(4, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.ones(4, 1, 1, 1, dtype=torch.float32)

    from tinyvllm.layers.attention import _blockwise_online_prefill_attention

    _blockwise_online_prefill_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        context,
        num_heads=1,
        head_dim=1,
        scale=1.0,
    )

    assert manager.ensure_calls == [[0], [1], [2], [3]]
    assert manager.future_calls == [
        {0, 1, 2, 3},
        {1, 2, 3},
        {2, 3},
        {3},
    ]
    assert manager.protected_calls == [set(), set(), set(), set()]


def test_blockwise_prefill_read_windows_hint_capacity_bounded_future_prefix_blocks():
    manager = _PlanOnlyManager()
    manager.gpu_blocks = 4
    manager.logical_to_slot.update({3: 3, 4: 4})
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1, 2, 3, 4]],
        kv_offload_prefill_chunk_starts=[5],
        kv_offload_prefill_chunk_ends=[6],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[4],
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k = torch.ones(1, 1, 1, dtype=torch.float32)
    v = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.ones(5, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.ones(5, 1, 1, 1, dtype=torch.float32)

    from tinyvllm.layers.attention import _blockwise_online_prefill_attention

    _blockwise_online_prefill_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        context,
        num_heads=1,
        head_dim=1,
        scale=1.0,
    )

    assert manager.ensure_calls == [[0], [1], [2], [3], [4]]
    assert manager.future_calls == [
        {0, 1, 2, 4},
        {1, 2, 3, 4},
        {2, 3, 4},
        {3, 4},
        {4},
    ]
    assert manager.protected_calls == [{4}, {4}, {4}, {4}, {4}]


def test_blockwise_prefill_gqa_does_not_materialize_repeated_kv_heads():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1]],
        kv_offload_prefill_chunk_starts=[2],
        kv_offload_prefill_chunk_ends=[3],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.ones(1, 4, 1, dtype=torch.float32)
    k = torch.ones(1, 2, 1, dtype=torch.float32)
    v = torch.ones(1, 2, 1, dtype=torch.float32)
    k_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)
    v_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)

    with patch.object(
        attention_mod,
        "_repeat_kv_for_gqa",
        side_effect=AssertionError("repeat_kv_for_gqa called"),
    ):
        out = attention_mod._blockwise_online_prefill_attention(
            q,
            k,
            v,
            k_cache,
            v_cache,
            context,
            num_heads=4,
            head_dim=1,
            scale=1.0,
        )

    assert out.shape == (1, 4, 1)


def test_blockwise_prefill_prefix_windows_do_not_zero_fill_dense_buffers():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1]],
        kv_offload_prefill_chunk_starts=[2],
        kv_offload_prefill_chunk_ends=[3],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.ones(1, 4, 1, dtype=torch.float32)
    k = torch.ones(1, 2, 1, dtype=torch.float32)
    v = torch.ones(1, 2, 1, dtype=torch.float32)
    k_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)
    v_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)

    def fail_new_zeros(*args, **kwargs):
        raise AssertionError("new_zeros called for fully-copied prefix window")

    with patch.object(q, "new_zeros", side_effect=fail_new_zeros):
        out = attention_mod._blockwise_online_prefill_attention(
            q,
            k,
            v,
            k_cache,
            v_cache,
            context,
            num_heads=4,
            head_dim=1,
            scale=1.0,
        )

    assert out.shape == (1, 4, 1)


def test_normalize_logical_block_rows_filters_once_and_reports_max_blocks():
    rows, max_blocks = _normalize_logical_block_rows([
        [2, -1, "3"],
        [],
        [4, 5, -1],
    ])

    assert rows == [[2, 3], [], [4, 5]]
    assert max_blocks == 2


def test_decode_window_mask_reuses_position_template():
    positions = torch.arange(4).view(1, 1, -1)
    mask = _decode_window_mask(
        window_lens=[2, 4],
        max_window_tokens=3,
        positions_template=positions,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([
        [[True, True, False]],
        [[True, True, True]],
    ])
    assert torch.equal(mask, expected)


def test_local_causal_mask_reuses_position_templates():
    q_positions = torch.arange(4).view(4, 1, 1)
    k_positions = torch.arange(4).view(1, 1, 4)
    mask = _local_causal_mask(
        q_len=3,
        q_positions_template=q_positions,
        k_positions_template=k_positions,
    )

    expected = torch.tensor([
        [[True, False, False]],
        [[True, True, False]],
        [[True, True, True]],
    ])
    assert torch.equal(mask, expected)


def test_merge_attention_window_accepts_none_mask_as_all_valid():
    running_m = torch.full((1, 1), float("-inf"))
    running_l = torch.zeros((1, 1))
    running_o = torch.zeros((1, 1, 1))
    scores = torch.tensor([[[1.0, 2.0]]])
    values = torch.tensor([[[3.0]], [[5.0]]])

    merged_m, merged_l, merged_o = _merge_attention_window(
        running_m,
        running_l,
        running_o,
        scores,
        values,
        mask=None,
    )

    expected_weights = torch.exp(scores - torch.tensor([[[2.0]]]))
    assert torch.equal(merged_m, torch.tensor([[2.0]]))
    assert torch.allclose(merged_l, expected_weights.sum(dim=-1))
    assert torch.allclose(merged_o, torch.einsum("qht,thd->qhd", expected_weights, values))


def test_merge_attention_window_none_mask_does_not_allocate_valid_mask():
    running_m = torch.full((1, 1), float("-inf"))
    running_l = torch.zeros((1, 1))
    running_o = torch.zeros((1, 1, 1))
    scores = torch.tensor([[[1.0, 2.0]]])
    values = torch.tensor([[[3.0]], [[5.0]]])

    original_ones = torch.ones
    torch.ones = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.ones called"))
    try:
        _merge_attention_window(running_m, running_l, running_o, scores, values, mask=None)
    finally:
        torch.ones = original_ones


def main():
    test_blockwise_decode_stages_read_window_in_first_seen_order()
    test_blockwise_decode_read_windows_hint_capacity_bounded_future_blocks()
    test_blockwise_decode_reuses_cached_read_window_plan_across_layers()
    test_blockwise_decode_gqa_does_not_materialize_repeated_kv_heads()
    test_gqa_grouped_helpers_match_repeated_kv_reference()
    test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks()
    test_blockwise_prefill_future_hint_blocks_fill_only_spare_capacity()
    test_blockwise_prefill_read_windows_hint_next_prefix_blocks()
    test_blockwise_prefill_read_windows_hint_capacity_bounded_future_prefix_blocks()
    test_blockwise_prefill_gqa_does_not_materialize_repeated_kv_heads()
    test_blockwise_prefill_prefix_windows_do_not_zero_fill_dense_buffers()
    test_normalize_logical_block_rows_filters_once_and_reports_max_blocks()
    test_decode_window_mask_reuses_position_template()
    test_local_causal_mask_reuses_position_templates()
    test_merge_attention_window_accepts_none_mask_as_all_valid()
    test_merge_attention_window_none_mask_does_not_allocate_valid_mask()
    print("blockwise attention planning tests passed")


if __name__ == "__main__":
    main()
