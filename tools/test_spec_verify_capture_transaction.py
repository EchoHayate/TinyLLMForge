"""Dependency-light tests for spec-verify capture scratch leases."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "spec_verify_exact_cuda_graph_cache.py"
)


def load_cache_module():
    spec = importlib.util.spec_from_file_location(
        "spec_verify_capture_transaction_under_test",
        CACHE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_pool(module, *, block_ids=range(100, 108), block_size=256):
    return module.SpecVerifyCaptureScratchPool(
        block_ids=tuple(block_ids),
        block_size=block_size,
    )


def test_required_scratch_capacity_covers_worst_terminal_block_offset():
    module = load_cache_module()
    required = module.required_spec_verify_capture_scratch_blocks

    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(),
        block_size=256,
    ) == 0
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(1, 8),
        block_size=256,
    ) == 8
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(256, 257),
        block_size=256,
    ) == 8
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(257, 258),
        block_size=256,
    ) == 12


def test_scratch_two_active_leases_are_disjoint_and_scheduler_invisible():
    module = load_cache_module()
    pool = make_pool(module)

    first = pool.acquire(
        active_batch_size=2,
        query_len=8,
        row_offsets=(0, 255),
    )
    second = pool.acquire(
        active_batch_size=2,
        query_len=8,
        row_offsets=(0, 255),
    )

    assert first.block_ids == (100, 101, 102)
    assert first.row_block_counts == (1, 2)
    assert second.block_ids == (103, 104, 105)
    assert second.row_block_counts == (1, 2)
    assert set(first.block_ids).isdisjoint(second.block_ids)
    assert min(first.block_ids + second.block_ids) >= 100
    source = inspect.getsource(module.SpecVerifyCaptureScratchPool)
    for forbidden in (
        "BlockManager",
        "Sequence",
        "hash_index",
        "refcount",
    ):
        assert forbidden not in source


def test_scratch_rollback_releases_all_blocks_and_advances_generations():
    module = load_cache_module()
    pool = make_pool(module, block_ids=range(100, 104))
    lease = pool.acquire(
        active_batch_size=2,
        query_len=257,
        row_offsets=(0, 255),
    )

    assert lease.block_ids == (100, 101, 102, 103)
    assert lease.block_generations == (0, 0, 0, 0)

    pool.rollback(lease)

    assert lease.state == "rolled_back"
    replacement = pool.acquire(
        active_batch_size=2,
        query_len=257,
        row_offsets=(0, 255),
    )
    assert replacement.block_ids == lease.block_ids
    assert replacement.block_generations == (1, 1, 1, 1)


def test_scratch_double_rollback_and_unknown_lease_fail_closed():
    module = load_cache_module()
    pool = make_pool(module)
    lease = pool.acquire(
        active_batch_size=1,
        query_len=1,
        row_offsets=(0,),
    )
    pool.rollback(lease)

    with pytest.raises(RuntimeError, match="rolled back"):
        pool.rollback(lease)

    unknown = module.SpecVerifyCaptureScratchLease(
        lease_id=999,
        block_ids=(100,),
        block_generations=(1,),
        row_block_counts=(1,),
    )
    with pytest.raises(ValueError, match="unknown"):
        pool.rollback(unknown)


def test_pool_exhaustion_reports_scratch_unavailable():
    module = load_cache_module()
    pool = make_pool(module, block_ids=(100,))
    pool.acquire(
        active_batch_size=1,
        query_len=1,
        row_offsets=(0,),
    )

    with pytest.raises(RuntimeError, match="scratch_unavailable"):
        pool.acquire(
            active_batch_size=1,
            query_len=1,
            row_offsets=(0,),
        )
