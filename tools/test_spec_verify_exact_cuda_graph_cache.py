"""Dependency-light tests for the speculative verifier CUDA Graph cache."""

from __future__ import annotations

import importlib.util
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
        "spec_verify_exact_cuda_graph_cache_under_test",
        CACHE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity_kwargs(**overrides):
    values = {
        "active_batch_size": 4,
        "query_len": 3,
        "total_query_tokens": 12,
        "page_table_width": 17,
        "flash_attn_num_splits": 16,
        "attention_backend": "flash_attn",
        "attention_backend_version": "2.7.4",
        "input_dtype": "torch.int64",
        "output_dtype": "torch.bfloat16",
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_block_size": 256,
        "device_compute_capability": (8, 0),
    }
    values.update(overrides)
    return values


def make_identity(module, **overrides):
    return module.SpecVerifyGraphIdentity(
        **identity_kwargs(**overrides)
    )


def make_config(module, **overrides):
    values = {
        "enabled": True,
        "batch_allowlist": (1, 4),
        "query_len_allowlist": (1, 3),
        "min_observations": 2,
        "max_entries": 8,
        "max_static_bytes": 64 * 1024 * 1024,
        "max_reserved_bytes": 512 * 1024 * 1024,
        "max_total_capture_ns": 5_000_000_000,
        "max_single_capture_ns": 2_000_000_000,
    }
    values.update(overrides)
    return module.SpecVerifyExactCudaGraphCacheConfig(**values)


def make_entry(
    module,
    identity,
    *,
    static_bytes=128,
    capture_duration_ns=100,
    reserved_delta_bytes=64,
    last_use_step=0,
):
    return module.SpecVerifyExactCudaGraphEntry(
        identity=identity,
        identity_sha256=identity.sha256,
        graph=object(),
        tensors={"outputs": object()},
        static_bytes=static_bytes,
        capture_duration_ns=capture_duration_ns,
        allocated_delta_bytes=32,
        reserved_delta_bytes=reserved_delta_bytes,
        last_use_step=last_use_step,
    )


def capture_ready_entry(
    module,
    cache,
    identity,
    *,
    step_id,
    static_bytes=128,
    capture_duration_ns=100,
    reserved_delta_bytes=64,
):
    first = cache.observe_success(
        identity,
        estimated_static_bytes=static_bytes,
        step_id=step_id,
    )
    assert first.should_capture is False
    decision = cache.observe_success(
        identity,
        estimated_static_bytes=static_bytes,
        step_id=step_id + 1,
    )
    assert decision.should_capture is True
    entry = make_entry(
        module,
        identity,
        static_bytes=static_bytes,
        capture_duration_ns=capture_duration_ns,
        reserved_delta_bytes=reserved_delta_bytes,
    )
    cache.commit_capture(entry)
    return entry


def test_identity_sha_is_exact_and_deterministic():
    module = load_cache_module()
    identity = make_identity(module)
    assert identity.sha256 == make_identity(module).sha256

    for overrides in (
        {
            "active_batch_size": 1,
            "total_query_tokens": 3,
        },
        {
            "query_len": 4,
            "total_query_tokens": 16,
        },
        {"page_table_width": 18},
        {"attention_backend_version": "2.7.5"},
        {"output_dtype": "torch.float16"},
        {"device_compute_capability": (9, 0)},
    ):
        changed = make_identity(module, **overrides)
        assert changed.sha256 != identity.sha256


@pytest.mark.parametrize(
    "overrides",
    (
        {"active_batch_size": 0},
        {"active_batch_size": True},
        {"query_len": 0},
        {"total_query_tokens": 11},
        {"page_table_width": 0},
        {"flash_attn_num_splits": 8},
        {"num_query_heads": 0},
        {"num_kv_heads": 0},
        {"head_dim": 0},
        {"page_block_size": 0},
        {"device_compute_capability": (8,)},
        {"device_compute_capability": (True, 0)},
        {"attention_backend": ""},
        {"input_dtype": ""},
    ),
)
def test_identity_rejects_invalid_exact_shape_or_backend(overrides):
    module = load_cache_module()
    with pytest.raises(ValueError):
        make_identity(module, **overrides)


def test_cache_config_accepts_batch_one_and_empty_query_allowlist():
    module = load_cache_module()
    config = make_config(
        module,
        batch_allowlist=(1, 4),
        query_len_allowlist=(),
    )
    assert config.batch_allowlist == (1, 4)
    assert config.query_len_allowlist == ()


@pytest.mark.parametrize(
    "overrides",
    (
        {"enabled": 1},
        {"batch_allowlist": ()},
        {"batch_allowlist": (0, 1)},
        {"batch_allowlist": (1, True)},
        {"batch_allowlist": (4, 1)},
        {"batch_allowlist": (1, 1, 4)},
        {"query_len_allowlist": (0,)},
        {"query_len_allowlist": (1, True)},
        {"query_len_allowlist": (3, 1)},
        {"min_observations": 0},
        {"max_entries": True},
        {"max_static_bytes": 0},
        {"max_reserved_bytes": 0},
        {"max_total_capture_ns": 0},
        {"max_single_capture_ns": 0},
    ),
)
def test_cache_config_rejects_noncanonical_or_nonpositive_values(overrides):
    module = load_cache_module()
    with pytest.raises(ValueError):
        make_config(module, **overrides)


def test_observation_threshold_two_requests_capture_after_eager_successes():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(make_config(module))
    identity = make_identity(module)

    first = cache.observe_success(
        identity,
        estimated_static_bytes=128,
        step_id=10,
    )
    second = cache.observe_success(
        identity,
        estimated_static_bytes=128,
        step_id=11,
    )

    assert first == module.SpecVerifyGraphAdmissionDecision(
        should_capture=False,
        cache_state="observing",
        decision="cold",
        fallback_reason="cold_identity",
        observation_count=1,
    )
    assert second == module.SpecVerifyGraphAdmissionDecision(
        should_capture=True,
        cache_state="capturing",
        decision="capture",
        fallback_reason=None,
        observation_count=2,
    )
    assert cache.summary()["capture_attempts"] == 1


def test_ready_entry_tracks_hit_replay_and_in_flight_lifecycle():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(make_config(module))
    identity = make_identity(module)
    entry = capture_ready_entry(
        module,
        cache,
        identity,
        step_id=1,
    )

    assert cache.ready_entry(identity) is entry
    cache.begin_replay(entry, step_id=7)
    assert entry.in_flight_replays == 1
    cache.finish_replay(entry, step_id=7, succeeded=True)

    assert entry.in_flight_replays == 0
    assert entry.replay_count == 1
    assert entry.last_replay_step == 7
    assert entry.last_use_step == 7
    summary = cache.summary()
    assert summary["hits"] == 1
    assert summary["ready_entries"] == (identity.sha256,)


def test_lru_evicts_only_ready_zero_in_flight_entry():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(
        make_config(
            module,
            max_entries=2,
            max_static_bytes=1024,
        )
    )
    first_identity = make_identity(module, query_len=1, total_query_tokens=4)
    second_identity = make_identity(module, query_len=2, total_query_tokens=8)
    third_identity = make_identity(module, query_len=3, total_query_tokens=12)
    first = capture_ready_entry(
        module,
        cache,
        first_identity,
        step_id=1,
    )
    second = capture_ready_entry(
        module,
        cache,
        second_identity,
        step_id=3,
    )
    first.last_use_step = 2
    second.last_use_step = 5
    cache.begin_replay(first, step_id=6)

    cache.observe_success(
        third_identity,
        estimated_static_bytes=128,
        step_id=7,
    )
    decision = cache.observe_success(
        third_identity,
        estimated_static_bytes=128,
        step_id=8,
    )

    assert decision.should_capture is True
    assert first_identity.sha256 in cache.ready_entries
    assert second_identity.sha256 not in cache.ready_entries
    assert cache.summary()["evictions"] == 1
    cache.finish_replay(first, step_id=9, succeeded=True)


def test_entry_limit_does_not_evict_in_flight_or_capturing_entries():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(
        make_config(module, max_entries=1)
    )
    first_identity = make_identity(module, query_len=1, total_query_tokens=4)
    second_identity = make_identity(module, query_len=2, total_query_tokens=8)
    first = capture_ready_entry(
        module,
        cache,
        first_identity,
        step_id=1,
    )
    cache.begin_replay(first, step_id=4)
    cache.observe_success(
        second_identity,
        estimated_static_bytes=128,
        step_id=5,
    )
    decision = cache.observe_success(
        second_identity,
        estimated_static_bytes=128,
        step_id=6,
    )

    assert decision.should_capture is False
    assert decision.fallback_reason == "entry_limit"
    assert second_identity.sha256 not in cache.quarantined
    cache.finish_replay(first, step_id=7, succeeded=True)


def test_quarantine_reason_is_stable_and_identity_never_recaptures():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(make_config(module))
    identity = make_identity(module)

    cache.quarantine(identity, "capture_failed")
    cache.quarantine(identity, "capture_failed")
    with pytest.raises(ValueError, match="cannot change"):
        cache.quarantine(identity, "replay_failed")

    decision = cache.observe_success(
        identity,
        estimated_static_bytes=128,
        step_id=1,
    )
    assert decision.should_capture is False
    assert decision.cache_state == "quarantined"
    assert decision.decision == "quarantined"
    assert decision.fallback_reason == "capture_failed"
    assert cache.summary()["quarantines"] == 1


@pytest.mark.parametrize(
    ("config_overrides", "entry_overrides", "expected_reason"),
    (
        (
            {"max_static_bytes": 64},
            {"static_bytes": 128},
            "static_byte_budget",
        ),
        (
            {"max_reserved_bytes": 32},
            {"reserved_delta_bytes": 64},
            "post_capture_budget",
        ),
        (
            {"max_single_capture_ns": 50},
            {"capture_duration_ns": 100},
            "post_capture_budget",
        ),
        (
            {"max_total_capture_ns": 50},
            {"capture_duration_ns": 100},
            "post_capture_budget",
        ),
    ),
)
def test_cache_budgets_reject_or_quarantine_without_ready_publish(
    config_overrides,
    entry_overrides,
    expected_reason,
):
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(
        make_config(module, **config_overrides)
    )
    identity = make_identity(module)
    first = cache.observe_success(
        identity,
        estimated_static_bytes=128,
        step_id=1,
    )
    decision = cache.observe_success(
        identity,
        estimated_static_bytes=128,
        step_id=2,
    )

    if expected_reason == "static_byte_budget":
        assert first.should_capture is False
        assert decision.should_capture is False
        assert decision.fallback_reason == expected_reason
        assert cache.ready_entry(identity) is None
        return

    assert decision.should_capture is True
    entry = make_entry(
        module,
        identity,
        **entry_overrides,
    )
    cache.commit_capture(entry)
    assert cache.ready_entry(identity) is None
    assert cache.quarantined[identity.sha256] == expected_reason


def test_eviction_retains_conservative_reserved_memory_accounting():
    module = load_cache_module()
    cache = module.SpecVerifyExactCudaGraphCache(
        make_config(module, max_entries=1)
    )
    first_identity = make_identity(module, query_len=1, total_query_tokens=4)
    second_identity = make_identity(module, query_len=2, total_query_tokens=8)
    capture_ready_entry(
        module,
        cache,
        first_identity,
        step_id=1,
        reserved_delta_bytes=96,
    )
    cache.observe_success(
        second_identity,
        estimated_static_bytes=128,
        step_id=3,
    )
    decision = cache.observe_success(
        second_identity,
        estimated_static_bytes=128,
        step_id=4,
    )

    assert decision.should_capture is True
    assert cache.summary()["reserved_delta_bytes"] == 96
