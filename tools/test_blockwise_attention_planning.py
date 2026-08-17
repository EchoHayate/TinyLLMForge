"""Blockwise attention planning regression tests.

Run on an environment with torch/CUDA available:
  PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import tinyvllm.layers.attention as attention_mod

from tinyvllm.layers.attention import (
    BlockwiseDecodePlan,
    _blockwise_online_decode_attention,
    _blockwise_prefill_future_hint_blocks,
    _blockwise_read_window_future_hint_blocks,
    _bounded_cross_layer_reuse_blocks,
    _build_blockwise_decode_window_plan,
    _decode_window_mask,
    _gqa_scores_decode,
    _gqa_scores_prefill,
    _gqa_weighted_values_decode,
    _gqa_weighted_values_prefill,
    _local_causal_mask,
    _merge_attention_window,
    _normalize_logical_block_rows,
    _stage_blockwise_read_window,
    _unique_blocks_in_order,
)


class _PlanOnlyManager:
    def __init__(self):
        self.gpu_blocks = 4
        self.logical_to_slot = {}
        self.stats = {
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
            "decode_plan_builds": 0,
            "decode_plan_cache_hits": 0,
            "decode_plan_identity_invalidations": 0,
            "decode_windows_with_spare_capacity": 0,
            "decode_cross_layer_hint_blocks": 0,
            "decode_cross_layer_hint_resident": 0,
            "decode_cross_layer_hint_retained": 0,
        }
        self.ensure_calls = []
        self.future_calls = []
        self.protected_calls = []
        self.wait_calls = []
        self.touch_calls = []
        self.pending_wait_blocks = set(range(128))
        self.clock = 0
        self.slot_last_used = [0] * 128

    def mark_dirty(self, blocks):
        pass

    def record_h2d_slot_read_window(self, **kwargs):
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
        mapping = {}
        for block in logical_blocks:
            block = int(block)
            self.logical_to_slot.setdefault(block, block)
            mapping[block] = self.logical_to_slot[block]
        return mapping

    def wait_for_blocks(self, logical_blocks, clear_pending=False):
        self.wait_calls.append((list(logical_blocks), bool(clear_pending)))
        if clear_pending:
            self.pending_wait_blocks.difference_update(int(block) for block in logical_blocks)

    def _touch(self, slot: int):
        self.touch_calls.append(int(slot))
        self.clock += 1
        if int(slot) >= len(self.slot_last_used):
            self.slot_last_used.extend([0] * (int(slot) + 1 - len(self.slot_last_used)))
        self.slot_last_used[int(slot)] = self.clock


class _ResidentPlanOnlyManager(_PlanOnlyManager):
    def __init__(self):
        super().__init__()
        self.logical_to_slot = {0: 0, 1: 1, 2: 2}
        self.pending_wait_blocks = set()


class _SimulatedResidencyManager:
    def __init__(self, gpu_blocks):
        self.gpu_blocks = int(gpu_blocks)
        self.logical_to_slot = {}
        self.slot_to_logical = [None] * self.gpu_blocks
        self.slot_last_used = [0] * self.gpu_blocks
        self.pending_wait_blocks = set()
        self.clock = 0
        self.required_trace = []
        self.stats = {
            "h2d_copies": 0,
            "evictions": 0,
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
            "decode_plan_builds": 0,
            "decode_plan_cache_hits": 0,
            "decode_plan_identity_invalidations": 0,
            "decode_windows_with_spare_capacity": 0,
            "decode_cross_layer_hint_blocks": 0,
            "decode_cross_layer_hint_resident": 0,
            "decode_cross_layer_hint_retained": 0,
        }

    def mark_dirty(self, blocks):
        return None

    def record_h2d_slot_read_window(self, **kwargs):
        return None

    def _touch(self, slot):
        self.clock += 1
        self.slot_last_used[int(slot)] = self.clock

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ):
        required = _unique_blocks_in_order(logical_blocks)
        self.required_trace.append(tuple(required))
        protected = set(required) | set(protected_logical_blocks or ())
        future = set(future_logical_blocks or ())
        for logical in required:
            if logical in self.logical_to_slot:
                self._touch(self.logical_to_slot[logical])
                continue
            candidates = [
                slot
                for slot, resident in enumerate(self.slot_to_logical)
                if resident not in protected
            ]
            free = next(
                (
                    slot
                    for slot, resident in enumerate(self.slot_to_logical)
                    if resident is None
                ),
                None,
            )
            slot = free
            if slot is None:
                slot = min(
                    candidates,
                    key=lambda item: (
                        self.slot_to_logical[item] in future,
                        self.slot_last_used[item],
                    ),
                )
                old = self.slot_to_logical[slot]
                del self.logical_to_slot[old]
                self.stats["evictions"] += 1
            self.logical_to_slot[logical] = slot
            self.slot_to_logical[slot] = logical
            self.stats["h2d_copies"] += 1
            self._touch(slot)
        return {
            logical: self.logical_to_slot[logical]
            for logical in required
        }

    def wait_for_blocks(self, logical_blocks, clear_pending=False):
        return None


class _BackingResidencyManager(_SimulatedResidencyManager):
    def __init__(
        self,
        gpu_blocks,
        k_cache,
        v_cache,
        logical_k_blocks,
        logical_v_blocks,
    ):
        super().__init__(gpu_blocks)
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.logical_k_blocks = logical_k_blocks
        self.logical_v_blocks = logical_v_blocks

    def install_resident(self, logical_block, slot):
        logical_block = int(logical_block)
        slot = int(slot)
        self.logical_to_slot[logical_block] = slot
        self.slot_to_logical[slot] = logical_block
        self.k_cache[slot].copy_(self.logical_k_blocks[logical_block])
        self.v_cache[slot].copy_(self.logical_v_blocks[logical_block])
        self._touch(slot)

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ):
        mapping = super().ensure_resident(
            logical_blocks,
            require_valid,
            future_logical_blocks=future_logical_blocks,
            protected_logical_blocks=protected_logical_blocks,
        )
        for logical_block, slot in mapping.items():
            self.k_cache[slot].copy_(
                self.logical_k_blocks[int(logical_block)]
            )
            self.v_cache[slot].copy_(
                self.logical_v_blocks[int(logical_block)]
            )
        return mapping


def _decode_fixture(
    *,
    block_rows,
    context_lens,
    gpu_blocks,
    window_blocks,
    write_blocks=None,
):
    manager = _PlanOnlyManager()
    manager.gpu_blocks = int(gpu_blocks)
    manager.pending_wait_blocks = set()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=block_rows,
        kv_offload_context_lens=context_lens,
        kv_offload_blockwise_blocks=window_blocks,
        kv_offload_write_blocks=list(write_blocks or []),
        kv_offload_decode_window_plan_cache=None,
    )
    batch = len(block_rows)
    logical_blocks = max(
        (int(block) for row in block_rows for block in row),
        default=-1,
    ) + 1
    q = torch.ones(batch, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(
        max(logical_blocks, gpu_blocks),
        1,
        1,
        1,
        dtype=torch.float32,
    )
    v_cache = torch.zeros_like(k_cache)
    return manager, context, q, k_cache, v_cache


def _repeat_kv_heads(kv, num_heads):
    if kv.shape[1] == num_heads:
        return kv
    return kv.repeat_interleave(
        num_heads // kv.shape[1],
        dim=1,
    )


def _dense_spec_verify_reference(
    q,
    logical_k,
    logical_v,
    context_lens,
    query_len,
    scale,
):
    outputs = []
    for row_index, context_len in enumerate(context_lens):
        row_q = q[
            row_index * query_len:
            (row_index + 1) * query_len
        ].float()
        row_k = logical_k[row_index][:context_len].float()
        row_v = logical_v[row_index][:context_len].float()
        repeated_k = _repeat_kv_heads(row_k, row_q.size(1))
        repeated_v = _repeat_kv_heads(row_v, row_q.size(1))
        scores = torch.einsum(
            "qhd,khd->qhk",
            row_q,
            repeated_k,
        ) * scale
        query_start = context_len - query_len
        mask = (
            torch.arange(context_len).view(1, 1, -1)
            <= torch.arange(
                query_start,
                context_len,
            ).view(query_len, 1, 1)
        )
        probs = torch.softmax(
            scores.masked_fill(~mask, float("-inf")),
            dim=-1,
        )
        outputs.append(torch.einsum(
            "qhk,khd->qhd",
            probs,
            repeated_v,
        ))
    return torch.cat(outputs).to(q.dtype)


def _blockwise_spec_verify_fixture(context_lens, query_len):
    torch.manual_seed(20260812 + len(context_lens) * 10 + query_len)
    block_size = 4
    gpu_blocks = 12
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    logical_k = [
        torch.randn(length, num_kv_heads, head_dim)
        for length in context_lens
    ]
    logical_v = [
        torch.randn(length, num_kv_heads, head_dim)
        for length in context_lens
    ]
    block_rows = []
    logical_k_blocks = {}
    logical_v_blocks = {}
    next_block = 0
    for row_k, row_v in zip(logical_k, logical_v):
        row_blocks = []
        for start in range(0, row_k.shape[0], block_size):
            logical_block = next_block
            next_block += 1
            row_blocks.append(logical_block)
            k_block = torch.zeros(
                block_size,
                num_kv_heads,
                head_dim,
            )
            v_block = torch.zeros_like(k_block)
            take = min(block_size, row_k.shape[0] - start)
            k_block[:take] = row_k[start:start + take]
            v_block[:take] = row_v[start:start + take]
            logical_k_blocks[logical_block] = k_block
            logical_v_blocks[logical_block] = v_block
        block_rows.append(row_blocks)

    k_cache = torch.zeros(
        gpu_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )
    v_cache = torch.zeros_like(k_cache)
    manager = _BackingResidencyManager(
        gpu_blocks,
        k_cache,
        v_cache,
        logical_k_blocks,
        logical_v_blocks,
    )
    write_blocks = [row[-1] for row in block_rows]
    for slot, logical_block in enumerate(write_blocks):
        manager.install_resident(logical_block, slot)
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=block_rows,
        kv_offload_context_lens=list(context_lens),
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=write_blocks,
        spec_verify_query_lens=tuple(
            query_len for _ in context_lens
        ),
        kv_offload_spec_verify_window_plan_cache=None,
        kv_offload_spec_verify_position_template_cache=None,
        kv_offload_spec_verify_window_mask_cache=None,
    )
    q = torch.randn(
        len(context_lens) * query_len,
        num_heads,
        head_dim,
    )
    return (
        manager,
        context,
        q,
        k_cache,
        v_cache,
        logical_k,
        logical_v,
        write_blocks,
    )


def test_blockwise_spec_verify_matches_dense_causal_reference():
    cases = (
        ((11,), 2),
        ((19,), 4),
        ((9, 13, 17, 21), 2),
        ((12, 16, 20, 24), 4),
    )
    for context_lens, query_len in cases:
        expected = None
        for layer_idx in (0, 1):
            (
                manager,
                context,
                q,
                k_cache,
                v_cache,
                logical_k,
                logical_v,
                write_blocks,
            ) = _blockwise_spec_verify_fixture(
                context_lens,
                query_len,
            )
            if expected is None:
                expected = _dense_spec_verify_reference(
                    q,
                    logical_k,
                    logical_v,
                    context_lens,
                    query_len,
                    scale=0.125,
                )
            actual = attention_mod._blockwise_online_spec_verify_attention(
                q,
                k_cache,
                v_cache,
                context,
                num_heads=4,
                head_dim=8,
                scale=0.125,
                layer_idx=layer_idx,
            )

            torch.testing.assert_close(
                actual.float(),
                expected.float(),
                rtol=2e-4,
                atol=2e-4,
            )
            assert manager.stats["h2d_copies"] > 0
            assert sum(len(row) for row in (
                context.kv_offload_logical_block_tables
            )) > len(write_blocks)
            assert set(write_blocks).issubset(
                manager.logical_to_slot
            )


def _plan_without_cross_layer_hints(plan):
    return BlockwiseDecodePlan(
        identity=plan.identity,
        forward_windows=tuple(
            replace(window, cross_layer_reuse_blocks=())
            for window in plan.forward_windows
        ),
        reverse_windows=tuple(
            replace(window, cross_layer_reuse_blocks=())
            for window in plan.reverse_windows
        ),
    )


def _run_simulated_decode_layers(plan):
    manager = _SimulatedResidencyManager(gpu_blocks=2)
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1, 2, 3, 4, 5]],
        kv_offload_context_lens=[6],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=plan,
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.arange(
        2,
        dtype=torch.float32,
    ).view(2, 1, 1, 1)
    v_cache = k_cache.clone()
    for layer_idx in range(4):
        _blockwise_online_decode_attention(
            q,
            k_cache,
            v_cache,
            context,
            num_heads=1,
            head_dim=1,
            scale=1.0,
            layer_idx=layer_idx,
        )
    return manager


def test_decode_plan_builds_forward_and_reverse_cross_layer_frontiers():
    plan = _build_blockwise_decode_window_plan(
        block_rows=[[0, 1, 2, 3, 4]],
        context_lens=[5],
        max_blocks=5,
        block_size=1,
        window_blocks=1,
        write_blocks=set(),
        gpu_blocks=3,
    )

    assert isinstance(plan, BlockwiseDecodePlan)
    assert [window.required_blocks for window in plan.forward_windows] == [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
    ]
    assert [window.required_blocks for window in plan.reverse_windows] == [
        (4,),
        (3,),
        (2,),
        (1,),
        (0,),
    ]
    assert plan.forward_windows[-1].cross_layer_reuse_blocks == (3, 2)
    assert plan.reverse_windows[-1].cross_layer_reuse_blocks == (1, 2)


def test_cross_layer_reuse_is_stable_deduplicated_and_spare_bounded():
    assert _bounded_cross_layer_reuse_blocks(
        candidate_blocks=[3, 3, 2, 1, 0],
        required_blocks=(4,),
        write_blocks={0},
        gpu_blocks=4,
    ) == (3, 2)


def test_cross_layer_reuse_is_empty_without_spare_capacity():
    assert _bounded_cross_layer_reuse_blocks(
        candidate_blocks=[1, 2, 3],
        required_blocks=(0, 4),
        write_blocks={5},
        gpu_blocks=3,
    ) == ()


def test_decode_plan_exact_identity_reuses_cache():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=3,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )
    cached_plan = context.kv_offload_decode_window_plan_cache
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )

    assert context.kv_offload_decode_window_plan_cache is cached_plan
    assert manager.stats["decode_plan_builds"] == 1
    assert manager.stats["decode_plan_cache_hits"] == 1


def test_decode_plan_identity_change_rebuilds_cache():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=3,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )
    first_plan = context.kv_offload_decode_window_plan_cache
    context.kv_offload_context_lens = [2]
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )

    assert context.kv_offload_decode_window_plan_cache is not first_plan
    assert manager.stats["decode_plan_builds"] == 2
    assert manager.stats["decode_plan_identity_invalidations"] == 1


def test_cross_layer_hints_are_future_only():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2, 3]],
        context_lens=[4],
        gpu_blocks=3,
        window_blocks=1,
        write_blocks=[7],
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )

    assert manager.ensure_calls[-1] == [3]
    assert 2 in manager.future_calls[-1]
    assert 2 not in manager.protected_calls[-1]
    assert manager.wait_calls[-1] == ([3], True)
    assert 2 not in manager.pending_wait_blocks


def test_zero_spare_capacity_matches_existing_alternating_future_sets():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=1,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )

    assert manager.ensure_calls == [[2], [1], [0]]
    assert manager.future_calls == [{2}, {1}, {0}]
    assert manager.stats["decode_cross_layer_hint_blocks"] == 0


def test_resident_fast_path_touches_only_required_blocks():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=3,
        window_blocks=1,
    )
    manager.logical_to_slot = {0: 0, 2: 2}
    plan = _build_blockwise_decode_window_plan(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        max_blocks=3,
        block_size=1,
        window_blocks=1,
        write_blocks=set(),
        gpu_blocks=3,
    )
    context.kv_offload_decode_window_plan_cache = BlockwiseDecodePlan(
        identity=plan.identity,
        forward_windows=(plan.forward_windows[0],),
        reverse_windows=(plan.reverse_windows[-1],),
    )
    assert 2 in context.kv_offload_decode_window_plan_cache.reverse_windows[
        0
    ].cross_layer_reuse_blocks

    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )

    assert manager.ensure_calls == []
    assert manager.wait_calls == []
    assert manager.touch_calls == [0]


def test_cross_layer_hints_preserve_required_trace_and_do_not_worsen_movement():
    candidate_plan = _build_blockwise_decode_window_plan(
        block_rows=[[0, 1, 2, 3, 4, 5]],
        context_lens=[6],
        max_blocks=6,
        block_size=1,
        window_blocks=1,
        write_blocks=set(),
        gpu_blocks=2,
    )
    baseline_plan = _plan_without_cross_layer_hints(candidate_plan)

    baseline = _run_simulated_decode_layers(baseline_plan)
    candidate = _run_simulated_decode_layers(candidate_plan)

    assert candidate.required_trace == baseline.required_trace
    assert candidate.stats["h2d_copies"] <= baseline.stats["h2d_copies"]
    assert candidate.stats["evictions"] <= baseline.stats["evictions"]


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
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {1, 2, 3, 4},
    ]
    assert manager.protected_calls == [set(), set(), set(), set(), set()]
    assert manager.wait_calls == [
        ([0], True),
        ([1], True),
        ([2], True),
        ([3], True),
        ([4], True),
    ]


def test_blockwise_decode_odd_layers_stage_read_windows_from_tail():
    manager = _PlanOnlyManager()
    manager.gpu_blocks = 1
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
        layer_idx=1,
    )

    assert manager.ensure_calls == [[4], [3], [2], [1], [0]]
    assert manager.wait_calls == [
        ([4], True),
        ([3], True),
        ([2], True),
        ([1], True),
        ([0], True),
    ]


def test_blockwise_decode_odd_layers_hint_reverse_future_blocks():
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
        layer_idx=1,
    )

    assert manager.future_calls == [
        {1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3},
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


def test_blockwise_decode_reuses_cached_position_template_across_layers():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [0, 1],
        ],
        kv_offload_context_lens=[2],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=None,
        kv_offload_decode_position_template_cache=None,
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)

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
        attention_mod.torch,
        "arange",
        side_effect=AssertionError("decode position template recomputed"),
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


def test_blockwise_decode_reuses_cached_window_masks_across_layers():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [0, 1],
        ],
        kv_offload_context_lens=[2],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=None,
        kv_offload_decode_position_template_cache=None,
        kv_offload_decode_window_mask_cache=None,
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)

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
        "_decode_window_mask",
        side_effect=AssertionError("decode window mask recomputed"),
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


def test_blockwise_decode_full_windows_skip_mask_construction():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[
            [0, 1],
        ],
        kv_offload_context_lens=[2],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=None,
        kv_offload_decode_position_template_cache=None,
        kv_offload_decode_window_mask_cache=None,
    )
    q = torch.ones(1, 1, 1, dtype=torch.float32)
    k_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.zeros(2, 1, 1, 1, dtype=torch.float32)

    with patch.object(
        attention_mod,
        "_decode_window_mask",
        side_effect=AssertionError("full decode window should not build mask"),
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


def test_stage_blockwise_read_window_skips_resident_non_pending_window():
    manager = _ResidentPlanOnlyManager()
    before_clock = manager.clock

    unique_blocks = _stage_blockwise_read_window(
        manager,
        logical_blocks=[0, 1, 0],
        future_extra_blocks={2},
        protected_extra_blocks=set(),
        capacity_extra_blocks=set(),
        capacity_error_prefix="unused",
    )

    assert unique_blocks == [0, 1]
    assert manager.stats["prefetch_plans"] == 0
    assert manager.stats["prefetch_read_blocks"] == 0
    assert manager.ensure_calls == []
    assert manager.wait_calls == []
    assert manager.clock == before_clock + 2
    assert manager.slot_last_used[0] == before_clock + 1
    assert manager.slot_last_used[1] == before_clock + 2


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


def test_blockwise_prefill_reuses_cached_prefix_window_plan_across_layers():
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
        kv_offload_prefill_window_plan_cache=None,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.ones(1, 4, 1, dtype=torch.float32)
    k = torch.ones(1, 2, 1, dtype=torch.float32)
    v = torch.ones(1, 2, 1, dtype=torch.float32)
    k_cache = torch.ones(5, 1, 2, 1, dtype=torch.float32)
    v_cache = torch.ones(5, 1, 2, 1, dtype=torch.float32)

    attention_mod._blockwise_online_prefill_attention(
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

    with patch.object(
        attention_mod,
        "_blockwise_prefill_future_hint_blocks",
        side_effect=AssertionError("prefill prefix-window plan recomputed"),
    ):
        attention_mod._blockwise_online_prefill_attention(
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


def test_blockwise_prefill_reuses_cached_local_position_templates_across_layers():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1]],
        kv_offload_prefill_chunk_starts=[1],
        kv_offload_prefill_chunk_ends=[3],
        kv_offload_blockwise_blocks=1,
        kv_offload_write_blocks=[],
        kv_offload_prefill_window_plan_cache=None,
        kv_offload_prefill_position_template_cache=None,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
    )
    q = torch.ones(2, 2, 1, dtype=torch.float32)
    k = torch.ones(2, 1, 1, dtype=torch.float32)
    v = torch.ones(2, 1, 1, dtype=torch.float32)
    k_cache = torch.ones(2, 1, 1, 1, dtype=torch.float32)
    v_cache = torch.ones(2, 1, 1, 1, dtype=torch.float32)

    attention_mod._blockwise_online_prefill_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        context,
        num_heads=2,
        head_dim=1,
        scale=1.0,
    )

    with patch.object(
        attention_mod.torch,
        "arange",
        side_effect=AssertionError("prefill local position templates recomputed"),
    ):
        attention_mod._blockwise_online_prefill_attention(
            q,
            k,
            v,
            k_cache,
            v_cache,
            context,
            num_heads=2,
            head_dim=1,
            scale=1.0,
        )


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


def test_blockwise_decode_full_windows_do_not_zero_fill_dense_buffers():
    manager = _PlanOnlyManager()
    context = SimpleNamespace(
        kv_offload_manager=manager,
        kv_offload_logical_block_tables=[[0, 1]],
        kv_offload_context_lens=[2],
        kv_offload_blockwise_blocks=2,
        kv_offload_write_blocks=[],
        kv_offload_decode_window_plan_cache=None,
        kv_offload_decode_position_template_cache=None,
    )
    q = torch.ones(1, 4, 1, dtype=torch.float32)
    k_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)
    v_cache = torch.ones(2, 1, 2, 1, dtype=torch.float32)

    def fail_new_zeros(*args, **kwargs):
        raise AssertionError("new_zeros called for fully-copied decode window")

    with patch.object(q, "new_zeros", side_effect=fail_new_zeros):
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
    test_decode_plan_builds_forward_and_reverse_cross_layer_frontiers()
    test_cross_layer_reuse_is_stable_deduplicated_and_spare_bounded()
    test_cross_layer_reuse_is_empty_without_spare_capacity()
    test_decode_plan_exact_identity_reuses_cache()
    test_decode_plan_identity_change_rebuilds_cache()
    test_cross_layer_hints_are_future_only()
    test_zero_spare_capacity_matches_existing_alternating_future_sets()
    test_resident_fast_path_touches_only_required_blocks()
    test_cross_layer_hints_preserve_required_trace_and_do_not_worsen_movement()
    test_blockwise_decode_stages_read_window_in_first_seen_order()
    test_blockwise_decode_read_windows_hint_capacity_bounded_future_blocks()
    test_blockwise_decode_odd_layers_stage_read_windows_from_tail()
    test_blockwise_decode_odd_layers_hint_reverse_future_blocks()
    test_blockwise_decode_reuses_cached_read_window_plan_across_layers()
    test_blockwise_decode_reuses_cached_position_template_across_layers()
    test_blockwise_decode_reuses_cached_window_masks_across_layers()
    test_blockwise_decode_full_windows_skip_mask_construction()
    test_blockwise_decode_gqa_does_not_materialize_repeated_kv_heads()
    test_gqa_grouped_helpers_match_repeated_kv_reference()
    test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks()
    test_stage_blockwise_read_window_skips_resident_non_pending_window()
    test_blockwise_prefill_future_hint_blocks_fill_only_spare_capacity()
    test_blockwise_prefill_read_windows_hint_next_prefix_blocks()
    test_blockwise_prefill_read_windows_hint_capacity_bounded_future_prefix_blocks()
    test_blockwise_prefill_reuses_cached_prefix_window_plan_across_layers()
    test_blockwise_prefill_reuses_cached_local_position_templates_across_layers()
    test_blockwise_prefill_gqa_does_not_materialize_repeated_kv_heads()
    test_blockwise_prefill_prefix_windows_do_not_zero_fill_dense_buffers()
    test_blockwise_decode_full_windows_do_not_zero_fill_dense_buffers()
    test_normalize_logical_block_rows_filters_once_and_reports_max_blocks()
    test_decode_window_mask_reuses_position_template()
    test_local_causal_mask_reuses_position_templates()
    test_merge_attention_window_accepts_none_mask_as_all_valid()
    test_merge_attention_window_none_mask_does_not_allocate_valid_mask()
    print("blockwise attention planning tests passed")


if __name__ == "__main__":
    main()
