import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)
cache_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache",
    "tinyvllm/engine/qwen35_hybrid_prefix_cache.py",
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35HybridPrefixKey = cache_module.Qwen35HybridPrefixKey
Qwen35HybridPrefixSnapshotCache = (
    cache_module.Qwen35HybridPrefixSnapshotCache
)


def _layout(dtype):
    return HybridStateLayout(tuple(
        component
        for layer_index in (0, 2)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (2, 3),
                dtype,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 2, 2),
                dtype,
            ),
        )
    ))


def _fixture(dtype=torch.float32, max_entries=4, max_bytes=1 << 20):
    pool = HybridStateTensorPool(_layout(dtype), capacity=4, device="cpu")
    leases = tuple(HybridStateLease(index, 1, 100 + index) for index in range(4))
    for lease in leases:
        pool.activate(lease)
    adapters = (
        Qwen35LayerStateAdapter(pool, 0),
        Qwen35LayerStateAdapter(pool, 2),
    )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    for layer_offset, adapter in enumerate(adapters):
        for slot_id in range(4):
            adapter.convolution[slot_id].copy_(
                torch.arange(6).reshape(2, 3)
                + layer_offset * 1000
                + slot_id * 100
            )
            adapter.recurrent[slot_id].copy_(
                torch.arange(8).reshape(2, 2, 2)
                + layer_offset * 2000
                + slot_id * 200
            )
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=max_entries,
        max_bytes=max_bytes,
    )
    return pool, leases, adapters, transaction, cache


def _key(
    *,
    token_hash=101,
    token_count=4,
    terminal_block_hash=201,
    block_size=4,
    model_fingerprint="model-a",
    layout_fingerprint="layout-a",
    tensor_parallel_size=1,
    dtype=torch.float32,
):
    return Qwen35HybridPrefixKey(
        token_hash=token_hash,
        token_count=token_count,
        terminal_block_hash=terminal_block_hash,
        block_size=block_size,
        model_fingerprint=model_fingerprint,
        layout_fingerprint=layout_fingerprint,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )


def _tokens(base=1):
    return tuple(range(base, base + 4))


def _blocks(block_id=7, generation=3, block_hash=201):
    return ((block_id, generation, block_hash),)


def _state_rows(adapters, slot_ids):
    return tuple(
        (
            adapter.convolution[list(slot_ids)].clone(),
            adapter.recurrent[list(slot_ids)].clone(),
        )
        for adapter in adapters
    )


def _cached_snapshots(cache):
    return tuple(cache._entries.values())


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_publish_clones_one_source_row_and_exact_acquire_broadcasts():
    _, leases, adapters, _, cache = _fixture()
    source = _state_rows(adapters, (0,))
    assert cache.publish(_key(), _tokens(), _blocks(), leases[0]) is True
    for adapter in adapters:
        adapter.convolution[0].add_(9000)
        adapter.recurrent[0].add_(9000)
        adapter.convolution[1:].zero_()
        adapter.recurrent[1:].zero_()

    assert cache.acquire(
        _key(),
        _tokens(),
        _blocks(),
        (leases[3], leases[1], leases[2]),
    ) is True
    for layer_index, adapter in enumerate(adapters):
        for slot_id in (3, 1, 2):
            torch.testing.assert_close(
                adapter.convolution[slot_id],
                source[layer_index][0][0],
            )
            torch.testing.assert_close(
                adapter.recurrent[slot_id],
                source[layer_index][1][0],
            )
    snapshot = cache.observation_snapshot()
    expected_bytes = sum(
        tensor[0].numel() * tensor.element_size()
        for pair in source
        for tensor in pair
    )
    assert snapshot["current_entries"] == 1
    assert snapshot["current_bytes"] == expected_bytes
    assert snapshot["hits"] == 1


def test_prepared_publication_is_invisible_and_rollback_preserves_entry():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    previous_snapshot = _cached_snapshots(cache)[0]
    before = cache.observation_snapshot()

    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[1],
    )
    assert prepared is not None
    during = cache.observation_snapshot()
    assert _cached_snapshots(cache) == (previous_snapshot,)
    assert during["current_entries"] == before["current_entries"]
    assert during["current_bytes"] == before["current_bytes"]
    assert during["current_logical_bytes"] == before[
        "current_logical_bytes"
    ]
    assert during["current_prepared_publications"] == 1
    assert during["current_prepared_bytes"] == previous_snapshot.storage_bytes
    assert during["publication_prepares"] == (
        before["publication_prepares"] + 1
    )

    for adapter in adapters:
        adapter.convolution[1].add_(9000)
        adapter.recurrent[1].add_(9000)
    cache.rollback_publication(prepared)
    after = cache.observation_snapshot()
    assert _cached_snapshots(cache) == (previous_snapshot,)
    assert after["current_prepared_publications"] == 0
    assert after["current_prepared_bytes"] == 0
    assert after["publication_rollbacks"] == (
        before["publication_rollbacks"] + 1
    )
    assert after["current_entries"] == before["current_entries"]
    assert after["current_bytes"] == before["current_bytes"]
    assert after["current_logical_bytes"] == before[
        "current_logical_bytes"
    ]

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True


def test_abort_current_publication_rolls_back_owned_prepared_state():
    _, leases, _, _, cache = _fixture()
    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    assert prepared is not None
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 1

    assert cache.abort_current_publication() is True
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0
    assert cache.abort_current_publication() is False


def test_prepared_publication_commit_uses_staged_values_and_consumes_handle():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    staged = _state_rows(adapters, (1,))
    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[1],
    )
    assert prepared is not None

    for adapter in adapters:
        adapter.convolution[1].add_(9000)
        adapter.recurrent[1].add_(9000)
    assert cache.commit_publication(prepared) is True
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 1
    assert observation["current_prepared_publications"] == 0
    assert observation["current_prepared_bytes"] == 0
    assert observation["publication_prepares"] == 1
    assert observation["publication_commits"] == 1

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[2],
            staged[layer_index][0][0],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            staged[layer_index][1][0],
        )

    _expect_error(
        lambda: cache.commit_publication(prepared),
        RuntimeError,
        "not current",
    )
    _expect_error(
        lambda: cache.rollback_publication(prepared),
        RuntimeError,
        "not current",
    )


def test_precommit_is_invisible_and_finalize_performs_no_intern_work():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    staged = _state_rows(adapters, (0,))
    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[0],
    )
    assert prepared is not None
    cache.precommit_publication(prepared)

    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 0
    assert observation["current_bytes"] == 0
    assert observation["current_logical_bytes"] == 0
    assert observation["current_precommitted_bytes"] == 112
    assert observation["current_precommitted_references"] == 4
    assert observation["publication_precommits"] == 1

    original_digest = cache_module._tensor_digest
    original_acquire = cache._acquire_interned_tensor
    digest_calls = 0
    acquire_calls = 0

    def counting_digest(tensor):
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(tensor)

    def counting_acquire(candidate, intern_key=None):
        nonlocal acquire_calls
        acquire_calls += 1
        return original_acquire(candidate, intern_key)

    cache_module._tensor_digest = counting_digest
    cache._acquire_interned_tensor = counting_acquire
    try:
        assert cache.finalize_publication(prepared) is True
    finally:
        cache_module._tensor_digest = original_digest
        cache._acquire_interned_tensor = original_acquire
    assert digest_calls == 0
    assert acquire_calls == 0

    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 1
    assert observation["current_bytes"] == 112
    assert observation["current_logical_bytes"] == 112
    assert observation["current_precommitted_bytes"] == 0
    assert observation["current_precommitted_references"] == 0

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[2],
            staged[layer_index][0][0],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            staged[layer_index][1][0],
        )


def test_precommitted_rollback_releases_private_intern_refs():
    _, leases, _, _, cache = _fixture()
    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    before = cache.observation_snapshot()
    cache.precommit_publication(prepared)
    assert cache.observation_snapshot()[
        "current_precommitted_references"
    ] == 4
    cache.rollback_publication(prepared)
    after = cache.observation_snapshot()
    assert after["current_entries"] == 0
    assert after["current_bytes"] == 0
    assert after["current_precommitted_bytes"] == 0
    assert after["current_precommitted_references"] == 0
    assert after["current_interned_tensors"] == 0
    for field in ("intern_hits", "intern_misses", "intern_collisions"):
        assert after[field] == before[field]


def test_finalized_publication_can_rollback_new_entry_before_seal():
    _, leases, _, _, cache = _fixture()
    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    cache.precommit_publication(prepared)
    assert cache.finalize_publication(prepared) is True
    assert cache.observation_snapshot()["current_entries"] == 1
    cache.rollback_publication(prepared)
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 0
    assert observation["current_bytes"] == 0
    assert observation["current_logical_bytes"] == 0
    assert observation["current_interned_tensors"] == 0


def test_finalized_replacement_rollback_restores_previous_snapshot():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    previous = _cached_snapshots(cache)[0]
    previous_values = tuple(
        (
            tensor.clone(),
            previous.recurrent_states[index].clone(),
        )
        for index, tensor in enumerate(previous.convolution_states)
    )

    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[1],
    )
    cache.precommit_publication(prepared)
    assert cache.finalize_publication(prepared) is True
    assert _cached_snapshots(cache)[0] is not previous
    cache.rollback_publication(prepared)
    assert _cached_snapshots(cache) == (previous,)

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[2],
            previous_values[layer_index][0],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            previous_values[layer_index][1],
        )


def test_finalized_byte_evictions_rollback_restores_exact_lru():
    _, leases, _, transaction, _ = _fixture()
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=224,
    )
    keys = (
        _key(),
        _key(token_hash=102, terminal_block_hash=202),
        _key(token_hash=103, terminal_block_hash=203),
    )
    tokens = (_tokens(), _tokens(10), _tokens(20))
    blocks = (
        _blocks(),
        _blocks(8, 1, 202),
        _blocks(9, 1, 203),
    )
    assert cache.publish(keys[0], tokens[0], blocks[0], leases[0])
    assert cache.publish(keys[1], tokens[1], blocks[1], leases[1])
    cache.acquire(keys[0], tokens[0], blocks[0], (leases[2],))
    before_keys = tuple(cache._entries)
    before = cache.observation_snapshot()

    for adapter in transaction.adapters:
        adapter.convolution[2].add_(5000)
        adapter.recurrent[2].add_(5000)
    prepared = cache.prepare_publication(
        keys[2],
        tokens[2],
        blocks[2],
        leases[2],
    )
    cache.precommit_publication(prepared)
    assert cache.finalize_publication(prepared) is True
    assert tuple(cache._entries) != before_keys
    cache.rollback_publication(prepared)

    after = cache.observation_snapshot()
    assert tuple(cache._entries) == before_keys
    assert after["current_entries"] == before["current_entries"]
    assert after["current_bytes"] == before["current_bytes"]
    assert after["current_logical_bytes"] == before[
        "current_logical_bytes"
    ]
    assert after["evictions"] == before["evictions"]
    assert after["byte_limit_evictions"] == before[
        "byte_limit_evictions"
    ]


def test_prepared_publication_rejects_conflict_foreign_and_oversize():
    _, leases, _, transaction, cache = _fixture()
    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    assert prepared is not None
    _expect_error(
        lambda: cache.prepare_publication(
            _key(token_hash=102, terminal_block_hash=202),
            _tokens(10),
            _blocks(8, 1, 202),
            leases[1],
        ),
        RuntimeError,
        "already prepared",
    )
    assert cache.observation_snapshot()[
        "publication_prepare_conflicts"
    ] == 1

    foreign = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=1 << 20,
    )
    _expect_error(
        lambda: foreign.commit_publication(prepared),
        RuntimeError,
        "not current",
    )
    cache.rollback_publication(prepared)

    oversize = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=1,
    )
    assert oversize.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is None
    observation = oversize.observation_snapshot()
    assert observation["current_prepared_publications"] == 0
    assert observation["current_prepared_bytes"] == 0
    assert observation["oversize_rejections"] == 1


def test_prepared_commit_failure_preserves_entry_and_allows_rollback():
    _, leases, _, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    previous_snapshot = _cached_snapshots(cache)[0]
    before = cache.observation_snapshot()
    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[1],
    )
    assert prepared is not None

    original_acquire = cache._acquire_interned_tensor
    calls = 0

    def failing_acquire(candidate, intern_key=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected staged commit failure")
        return original_acquire(candidate, intern_key)

    cache._acquire_interned_tensor = failing_acquire
    _expect_error(
        lambda: cache.commit_publication(prepared),
        RuntimeError,
        "staged commit failure",
    )
    cache._acquire_interned_tensor = original_acquire

    after_failure = cache.observation_snapshot()
    assert _cached_snapshots(cache) == (previous_snapshot,)
    assert after_failure["current_prepared_publications"] == 1
    assert after_failure["current_prepared_bytes"] == prepared.storage_bytes
    for field in (
        "current_entries",
        "current_bytes",
        "current_logical_bytes",
        "current_interned_tensors",
        "current_intern_references",
        "publishes",
        "replacements",
        "intern_hits",
        "intern_misses",
        "intern_collisions",
        "publication_commits",
    ):
        assert after_failure[field] == before[field], (
            field,
            before,
            after_failure,
        )
    cache.rollback_publication(prepared)
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0


def test_exact_tensor_interning_shares_storage_and_releases_last_reference():
    _, leases, adapters, _, cache = _fixture()
    key_a = _key()
    tokens_a = _tokens()
    blocks_a = _blocks()
    key_b = _key(token_hash=102, terminal_block_hash=202)
    tokens_b = _tokens(10)
    blocks_b = _blocks(8, 1, 202)
    source = _state_rows(adapters, (0,))

    assert cache.publish(key_a, tokens_a, blocks_a, leases[0]) is True
    assert cache.publish(key_b, tokens_b, blocks_b, leases[0]) is True

    snapshots = _cached_snapshots(cache)
    assert len(snapshots) == 2
    for states_name in ("convolution_states", "recurrent_states"):
        states_a = getattr(snapshots[0], states_name)
        states_b = getattr(snapshots[1], states_name)
        for tensor_a, tensor_b in zip(states_a, states_b):
            assert tensor_a is tensor_b
            assert tensor_a.data_ptr() == tensor_b.data_ptr()

    expected_bytes = sum(
        tensor[0].numel() * tensor.element_size()
        for pair in source
        for tensor in pair
    )
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 2
    assert observation["current_bytes"] == expected_bytes
    assert observation["current_logical_bytes"] == expected_bytes * 2
    assert observation["deduplicated_bytes"] == expected_bytes
    assert observation["current_interned_tensors"] == 4
    assert observation["current_intern_references"] == 8

    for adapter in adapters:
        adapter.convolution[0].add_(9000)
        adapter.recurrent[0].add_(9000)
    for snapshot in snapshots:
        for cached, expected in zip(
            snapshot.convolution_states,
            (pair[0][0] for pair in source),
        ):
            torch.testing.assert_close(cached, expected)
        for cached, expected in zip(
            snapshot.recurrent_states,
            (pair[1][0] for pair in source),
        ):
            torch.testing.assert_close(cached, expected)

    assert cache.invalidate_blocks(blocks_a) == 1
    observation = cache.observation_snapshot()
    assert observation["current_bytes"] == expected_bytes
    assert observation["current_logical_bytes"] == expected_bytes
    assert observation["current_interned_tensors"] == 4
    assert observation["current_intern_references"] == 4

    assert cache.invalidate_blocks(blocks_b) == 1
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 0
    assert observation["current_bytes"] == 0
    assert observation["current_logical_bytes"] == 0
    assert observation["deduplicated_bytes"] == 0
    assert observation["current_interned_tensors"] == 0
    assert observation["current_intern_references"] == 0


def test_partial_tensor_interning_and_replacement_refcounts():
    _, leases, adapters, _, cache = _fixture()
    key_a = _key()
    tokens_a = _tokens()
    blocks_a = _blocks()
    key_b = _key(token_hash=102, terminal_block_hash=202)
    tokens_b = _tokens(10)
    blocks_b = _blocks(8, 1, 202)

    assert cache.publish(key_a, tokens_a, blocks_a, leases[0]) is True
    for adapter in adapters:
        adapter.convolution[1].copy_(adapter.convolution[0])
        adapter.recurrent[1].copy_(adapter.recurrent[0])
    adapters[1].recurrent[1].add_(1)
    assert cache.publish(key_b, tokens_b, blocks_b, leases[1]) is True

    full_snapshot_bytes = 112
    changed_tensor_bytes = adapters[1].recurrent[1].numel() * 4
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 2
    assert observation["current_bytes"] == (
        full_snapshot_bytes + changed_tensor_bytes
    )
    assert observation["current_logical_bytes"] == full_snapshot_bytes * 2
    assert observation["deduplicated_bytes"] == (
        full_snapshot_bytes - changed_tensor_bytes
    )
    assert observation["current_interned_tensors"] == 5
    assert observation["current_intern_references"] == 8

    adapters[1].recurrent[1].copy_(adapters[1].recurrent[0])
    assert cache.publish(key_b, tokens_b, blocks_b, leases[1]) is True
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 2
    assert observation["current_bytes"] == full_snapshot_bytes
    assert observation["current_logical_bytes"] == full_snapshot_bytes * 2
    assert observation["current_interned_tensors"] == 4
    assert observation["current_intern_references"] == 8
    assert observation["replacements"] == 1


def test_intern_failure_rolls_back_refs_and_preserves_previous_entry():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    before = cache.observation_snapshot()
    previous_snapshot = _cached_snapshots(cache)[0]

    original_acquire = cache._acquire_interned_tensor
    calls = 0

    def failing_acquire(candidate, key=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected intern failure")
        return original_acquire(candidate, key)

    cache._acquire_interned_tensor = failing_acquire
    _expect_error(
        lambda: cache.publish(key, tokens, blocks, leases[1]),
        RuntimeError,
        "intern failure",
    )
    cache._acquire_interned_tensor = original_acquire

    after = cache.observation_snapshot()
    for field in (
        "current_entries",
        "current_bytes",
        "current_logical_bytes",
        "deduplicated_bytes",
        "current_interned_tensors",
        "current_intern_references",
        "replacements",
        "publishes",
        "intern_hits",
        "intern_misses",
        "intern_collisions",
    ):
        assert after[field] == before[field], (field, before, after)
    assert _cached_snapshots(cache)[0] is previous_snapshot

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[2],
            previous_snapshot.convolution_states[layer_index],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            previous_snapshot.recurrent_states[layer_index],
        )


def test_publish_hashes_each_candidate_tensor_once():
    _, leases, _, _, cache = _fixture()
    original_digest = cache_module._tensor_digest
    digest_calls = 0

    def counting_digest(tensor):
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(tensor)

    cache_module._tensor_digest = counting_digest
    try:
        assert cache.publish(
            _key(),
            _tokens(),
            _blocks(),
            leases[0],
        ) is True
    finally:
        cache_module._tensor_digest = original_digest
    assert digest_calls == 4


def test_digest_collision_requires_exact_equality_and_restores_each_value():
    _, leases, adapters, _, cache = _fixture()
    original_digest = cache_module._tensor_digest
    cache_module._tensor_digest = lambda _: "forced-collision"
    try:
        key_a = _key()
        tokens_a = _tokens()
        blocks_a = _blocks()
        key_b = _key(token_hash=102, terminal_block_hash=202)
        tokens_b = _tokens(10)
        blocks_b = _blocks(8, 1, 202)
        key_c = _key(token_hash=103, terminal_block_hash=203)
        tokens_c = _tokens(20)
        blocks_c = _blocks(9, 1, 203)
        source_a = _state_rows(adapters, (0,))
        source_b = _state_rows(adapters, (1,))

        assert cache.publish(key_a, tokens_a, blocks_a, leases[0]) is True
        assert cache.publish(key_b, tokens_b, blocks_b, leases[1]) is True
        assert cache.publish(key_c, tokens_c, blocks_c, leases[1]) is True

        snapshots = _cached_snapshots(cache)
        for states_name in ("convolution_states", "recurrent_states"):
            states_a = getattr(snapshots[0], states_name)
            states_b = getattr(snapshots[1], states_name)
            states_c = getattr(snapshots[2], states_name)
            for tensor_a, tensor_b, tensor_c in zip(
                states_a,
                states_b,
                states_c,
            ):
                assert tensor_a.data_ptr() != tensor_b.data_ptr()
                assert tensor_b is tensor_c
        observation = cache.observation_snapshot()
        assert observation["current_bytes"] == 224
        assert observation["current_logical_bytes"] == 336
        assert observation["intern_collisions"] == 6

        for adapter in adapters:
            adapter.convolution[2:].zero_()
            adapter.recurrent[2:].zero_()
        assert cache.acquire(
            key_a,
            tokens_a,
            blocks_a,
            (leases[2],),
        ) is True
        assert cache.acquire(
            key_b,
            tokens_b,
            blocks_b,
            (leases[3],),
        ) is True
        for layer_index, adapter in enumerate(adapters):
            torch.testing.assert_close(
                adapter.convolution[2],
                source_a[layer_index][0][0],
            )
            torch.testing.assert_close(
                adapter.recurrent[2],
                source_a[layer_index][1][0],
            )
            torch.testing.assert_close(
                adapter.convolution[3],
                source_b[layer_index][0][0],
            )
            torch.testing.assert_close(
                adapter.recurrent[3],
                source_b[layer_index][1][0],
            )
    finally:
        cache_module._tensor_digest = original_digest


def test_digest_collision_does_not_alias_distinct_signed_zero_bytes():
    _, leases, adapters, _, cache = _fixture()
    for adapter in adapters:
        adapter.convolution.zero_()
        adapter.recurrent.zero_()
    adapters[0].convolution[1].fill_(-0.0)

    original_digest = cache_module._tensor_digest
    cache_module._tensor_digest = lambda _: "forced-collision"
    try:
        key_a = _key()
        key_b = _key(token_hash=102, terminal_block_hash=202)
        assert cache.publish(
            key_a,
            _tokens(),
            _blocks(),
            leases[0],
        ) is True
        assert cache.publish(
            key_b,
            _tokens(10),
            _blocks(8, 1, 202),
            leases[1],
        ) is True

        snapshots = _cached_snapshots(cache)
        positive_zero = snapshots[0].convolution_states[0]
        negative_zero = snapshots[1].convolution_states[0]
        assert torch.equal(positive_zero, negative_zero)
        assert positive_zero.data_ptr() != negative_zero.data_ptr()
        assert not torch.equal(
            positive_zero.view(torch.uint8),
            negative_zero.view(torch.uint8),
        )
        observation = cache.observation_snapshot()
        assert observation["current_bytes"] == 80
        assert observation["current_logical_bytes"] == 224
    finally:
        cache_module._tensor_digest = original_digest


def test_constructor_and_public_identity_validation():
    _, _, _, transaction, _ = _fixture()
    for kwargs, message in (
        ({"max_entries": 0, "max_bytes": 100}, "max_entries"),
        ({"max_entries": 1, "max_bytes": 0}, "max_bytes"),
    ):
        _expect_error(
            lambda kwargs=kwargs: Qwen35HybridPrefixSnapshotCache(
                transaction,
                **kwargs,
            ),
            ValueError,
            message,
        )
    _expect_error(
        lambda: Qwen35HybridPrefixSnapshotCache(
            "not a transaction",
            max_entries=1,
            max_bytes=100,
        ),
        ValueError,
        "state_transaction",
    )


def test_identity_misses_and_stale_blocks_do_not_mutate_destinations():
    _, leases, adapters, _, cache = _fixture()
    cache.publish(_key(), _tokens(), _blocks(), leases[0])
    original = _state_rows(adapters, (1, 2))
    misses = (
        (_key(), _tokens(9), _blocks()),
        (_key(model_fingerprint="model-b"), _tokens(), _blocks()),
        (_key(layout_fingerprint="layout-b"), _tokens(), _blocks()),
        (_key(tensor_parallel_size=2), _tokens(), _blocks()),
        (_key(dtype=torch.bfloat16), _tokens(), _blocks()),
        (
            _key(block_size=2),
            _tokens(),
            ((6, 1, 150), (7, 3, 201)),
        ),
        (_key(), _tokens(), _blocks(generation=4)),
        (_key(), _tokens(), _blocks(block_hash=202)),
    )
    for key, tokens, blocks in misses:
        assert cache.acquire(key, tokens, blocks, (leases[1], leases[2])) is False
        current = _state_rows(adapters, (1, 2))
        for current_pair, original_pair in zip(current, original):
            torch.testing.assert_close(current_pair[0], original_pair[0])
            torch.testing.assert_close(current_pair[1], original_pair[1])


def test_publish_validation_and_failure_preserve_previous_entry():
    _, leases, adapters, transaction, cache = _fixture()
    cache.publish(_key(), _tokens(), _blocks(), leases[0])
    invalid_cases = (
        (lambda: cache.publish(_key(token_count=3), (1, 2, 3), _blocks(), leases[0]), "block"),
        (lambda: cache.publish(_key(token_hash=-1), _tokens(), _blocks(), leases[0]), "token_hash"),
        (lambda: cache.publish(_key(model_fingerprint=""), _tokens(), _blocks(), leases[0]), "fingerprint"),
        (lambda: cache.publish(_key(), _tokens(), ((7, 3, 201), (7, 4, 202)), leases[0]), "block"),
    )
    for function, message in invalid_cases:
        _expect_error(function, ValueError, message)

    original_gather = transaction.gather

    def failing_gather(_):
        raise RuntimeError("injected gather failure")

    transaction.gather = failing_gather
    _expect_error(
        lambda: cache.publish(_key(), _tokens(), _blocks(), leases[1]),
        RuntimeError,
        "gather failure",
    )
    transaction.gather = original_gather
    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(_key(), _tokens(), _blocks(), (leases[2],)) is True


def test_restore_validation_and_late_copy_failure_are_atomic():
    _, leases, adapters, _, cache = _fixture()
    cache.publish(_key(), _tokens(), _blocks(), leases[0])
    entry_key = next(iter(cache._entries))
    snapshot = cache._entries[entry_key]
    original_recurrent = snapshot.recurrent_states[1]
    malformed_recurrent = (
        snapshot.recurrent_states[0],
        original_recurrent[:, :, :1],
    )
    cache._entries[entry_key] = replace(
        snapshot,
        recurrent_states=malformed_recurrent,
    )
    original = _state_rows(adapters, (1, 2))
    _expect_error(
        lambda: cache.acquire(_key(), _tokens(), _blocks(), (leases[1], leases[2])),
        ValueError,
        "recurrent",
    )
    cache._entries[entry_key] = snapshot
    current = _state_rows(adapters, (1, 2))
    for current_pair, original_pair in zip(current, original):
        torch.testing.assert_close(current_pair[0], original_pair[0])
        torch.testing.assert_close(current_pair[1], original_pair[1])

    calls = []
    original_copy = adapters[1]._copy_component

    def failing_copy(destination, source):
        calls.append(destination)
        if len(calls) == 3:
            raise RuntimeError("injected restore failure")
        return original_copy(destination, source)

    adapters[1]._copy_component = failing_copy
    _expect_error(
        lambda: cache.acquire(_key(), _tokens(), _blocks(), (leases[1], leases[2])),
        RuntimeError,
        "restore failure",
    )
    assert len(calls) == 3
    current = _state_rows(adapters, (1, 2))
    for current_pair, original_pair in zip(current, original):
        torch.testing.assert_close(current_pair[0], original_pair[0])
        torch.testing.assert_close(current_pair[1], original_pair[1])
    assert cache.observation_snapshot()["failed_restores"] == 1


def test_failed_restore_does_not_refresh_lru_recency():
    _, leases, adapters, _, cache = _fixture(max_entries=2)
    key_a = _key()
    tokens_a = _tokens()
    blocks_a = _blocks()
    key_b = _key(token_hash=102, terminal_block_hash=202)
    tokens_b = _tokens(10)
    blocks_b = _blocks(8, 1, 202)
    key_c = _key(token_hash=103, terminal_block_hash=203)
    tokens_c = _tokens(20)
    blocks_c = _blocks(9, 1, 203)
    cache.publish(key_a, tokens_a, blocks_a, leases[0])
    cache.publish(key_b, tokens_b, blocks_b, leases[0])

    original_copy = adapters[1]._copy_component

    def failing_copy(destination, source):
        raise RuntimeError("injected recency failure")

    adapters[1]._copy_component = failing_copy
    _expect_error(
        lambda: cache.acquire(key_a, tokens_a, blocks_a, (leases[1],)),
        RuntimeError,
        "recency failure",
    )
    adapters[1]._copy_component = original_copy
    cache.publish(key_c, tokens_c, blocks_c, leases[0])

    assert cache.acquire(key_a, tokens_a, blocks_a, (leases[1],)) is False
    assert cache.acquire(key_b, tokens_b, blocks_b, (leases[1],)) is True


def test_lru_entry_byte_limits_replacement_invalidation_and_clear():
    _, leases, _, _, cache = _fixture(max_entries=2)
    cache.publish(_key(), _tokens(), _blocks(), leases[0])
    first_bytes = cache.observation_snapshot()["current_bytes"]
    cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        leases[0],
    )
    cache.acquire(_key(), _tokens(), _blocks(), (leases[1],))
    cache.publish(
        _key(token_hash=103, terminal_block_hash=203),
        _tokens(20),
        _blocks(9, 1, 203),
        leases[0],
    )
    assert cache.acquire(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        (leases[1],),
    ) is False
    assert cache.acquire(_key(), _tokens(), _blocks(), (leases[1],)) is True

    cache.publish(_key(), _tokens(), _blocks(), leases[2])
    assert cache.observation_snapshot()["current_bytes"] == first_bytes * 2
    assert cache.observation_snapshot()["replacements"] == 1
    assert cache.invalidate_blocks(_blocks()) == 1
    assert cache.clear() == 1
    snapshot = cache.observation_snapshot()
    assert snapshot["current_entries"] == 0
    assert snapshot["current_bytes"] == 0
    assert snapshot["evictions"] == 1
    assert snapshot["entry_limit_evictions"] == 1
    assert snapshot["invalidations"] == 1
    assert snapshot["clears"] == 1
    assert snapshot["publishes"] == 4
    assert snapshot["hits"] == 2
    assert snapshot["misses"] == 1

    _, leases, _, _, byte_cache = _fixture(
        max_entries=4,
        max_bytes=first_bytes,
    )
    assert byte_cache.publish(_key(), _tokens(), _blocks(), leases[0]) is True
    assert byte_cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        leases[0],
    ) is True
    byte_snapshot = byte_cache.observation_snapshot()
    assert byte_snapshot["current_entries"] == 2
    assert byte_snapshot["current_bytes"] == first_bytes
    assert byte_snapshot["current_logical_bytes"] == first_bytes * 2
    assert byte_snapshot["deduplicated_bytes"] == first_bytes
    assert byte_snapshot["evictions"] == 0

    assert byte_cache.publish(
        _key(token_hash=103, terminal_block_hash=203),
        _tokens(20),
        _blocks(9, 1, 203),
        leases[1],
    ) is True
    byte_snapshot = byte_cache.observation_snapshot()
    assert byte_snapshot["current_entries"] == 1
    assert byte_snapshot["evictions"] == 2
    assert byte_snapshot["byte_limit_evictions"] == 2

    _, leases, _, _, oversize = _fixture(
        max_entries=4,
        max_bytes=first_bytes - 1,
    )
    assert oversize.publish(_key(), _tokens(), _blocks(), leases[0]) is False
    oversize_snapshot = oversize.observation_snapshot()
    assert oversize_snapshot["current_entries"] == 0
    assert oversize_snapshot["oversize_rejections"] == 1


def test_oversize_replacement_preserves_smaller_unique_snapshot():
    _, leases, adapters, transaction, _ = _fixture()
    for component_name in ("convolution", "recurrent"):
        getattr(adapters[1], component_name)[0].copy_(
            getattr(adapters[0], component_name)[0]
        )
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=80,
    )
    key = _key()
    tokens = _tokens()
    blocks = _blocks()

    assert cache.publish(key, tokens, blocks, leases[0]) is True
    before = cache.observation_snapshot()
    assert before["current_bytes"] == 56
    assert before["current_logical_bytes"] == 112

    assert cache.publish(key, tokens, blocks, leases[1]) is False
    after = cache.observation_snapshot()
    assert after["current_entries"] == 1
    assert after["current_bytes"] == 56
    assert after["current_logical_bytes"] == 112
    assert after["replacements"] == 0
    assert after["oversize_rejections"] == 1

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(key, tokens, blocks, (leases[2],)) is True
    for adapter in adapters:
        torch.testing.assert_close(
            adapter.convolution[2],
            adapters[0].convolution[0],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            adapters[0].recurrent[0],
        )


def test_bfloat16_storage_and_restore():
    _, leases, adapters, _, cache = _fixture(torch.bfloat16)
    source = _state_rows(adapters, (0,))
    assert cache.publish(
        _key(dtype=torch.bfloat16),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    for adapter in adapters:
        adapter.convolution[3].zero_()
        adapter.recurrent[3].zero_()
    assert cache.acquire(
        _key(dtype=torch.bfloat16),
        _tokens(),
        _blocks(),
        (leases[3],),
    ) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[3],
            source[layer_index][0][0],
        )
        torch.testing.assert_close(
            adapter.recurrent[3],
            source[layer_index][1][0],
        )


def test_fp32_pool_preserves_recurrent_state_at_bfloat16_snapshot_boundary():
    _, leases, adapters, _, cache = _fixture(torch.float32)
    expected_convolution = []
    expected_recurrent = []
    for layer_index, adapter in enumerate(adapters):
        convolution = (
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
            * 0.0013
            + layer_index * 0.0007
        )
        recurrent = (
            torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
            * 0.0011
            + layer_index * 0.0009
        )
        adapter.convolution[0].copy_(convolution)
        adapter.recurrent[0].copy_(recurrent)
        expected_convolution.append(convolution.clone())
        expected_recurrent.append(recurrent.clone())

    key = _key(dtype=torch.bfloat16)
    assert cache.publish(
        key,
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    snapshot = _cached_snapshots(cache)[0]
    for layer_index in range(len(adapters)):
        torch.testing.assert_close(
            snapshot.convolution_states[layer_index],
            expected_convolution[layer_index],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            snapshot.recurrent_states[layer_index],
            expected_recurrent[layer_index],
            rtol=0.0,
            atol=0.0,
        )

    for adapter in adapters:
        adapter.convolution[3].zero_()
        adapter.recurrent[3].zero_()
    assert cache.acquire(
        key,
        _tokens(),
        _blocks(),
        (leases[3],),
    ) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[3],
            expected_convolution[layer_index],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            adapter.recurrent[3],
            expected_recurrent[layer_index],
            rtol=0.0,
            atol=0.0,
        )


def main():
    test_publish_clones_one_source_row_and_exact_acquire_broadcasts()
    test_prepared_publication_is_invisible_and_rollback_preserves_entry()
    test_abort_current_publication_rolls_back_owned_prepared_state()
    test_prepared_publication_commit_uses_staged_values_and_consumes_handle()
    test_precommit_is_invisible_and_finalize_performs_no_intern_work()
    test_precommitted_rollback_releases_private_intern_refs()
    test_finalized_publication_can_rollback_new_entry_before_seal()
    test_finalized_replacement_rollback_restores_previous_snapshot()
    test_finalized_byte_evictions_rollback_restores_exact_lru()
    test_prepared_publication_rejects_conflict_foreign_and_oversize()
    test_prepared_commit_failure_preserves_entry_and_allows_rollback()
    test_exact_tensor_interning_shares_storage_and_releases_last_reference()
    test_partial_tensor_interning_and_replacement_refcounts()
    test_intern_failure_rolls_back_refs_and_preserves_previous_entry()
    test_publish_hashes_each_candidate_tensor_once()
    test_digest_collision_requires_exact_equality_and_restores_each_value()
    test_digest_collision_does_not_alias_distinct_signed_zero_bytes()
    test_constructor_and_public_identity_validation()
    test_identity_misses_and_stale_blocks_do_not_mutate_destinations()
    test_publish_validation_and_failure_preserve_previous_entry()
    test_restore_validation_and_late_copy_failure_are_atomic()
    test_failed_restore_does_not_refresh_lru_recency()
    test_lru_entry_byte_limits_replacement_invalidation_and_clear()
    test_oversize_replacement_preserves_smaller_unique_snapshot()
    test_bfloat16_storage_and_restore()
    test_fp32_pool_preserves_recurrent_state_at_bfloat16_snapshot_boundary()
    print("qwen35 hybrid prefix snapshot cache tests passed")


if __name__ == "__main__":
    main()
