import importlib
import gc
from dataclasses import replace
from pathlib import Path
import sys
import types
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid = importlib.import_module("tinyvllm.engine.hybrid_state")
adapter_module = importlib.import_module(
    "tinyvllm.engine.qwen35_layer_state"
)
transaction_module = importlib.import_module(
    "tinyvllm.engine.qwen35_state_transaction"
)
exact_cache_module = importlib.import_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache"
)
codec_module = importlib.import_module(
    "tinyvllm.engine.qwen35_recurrent_int8_codec"
)
representation_module = importlib.import_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_representation"
)
int8_cache_module = importlib.import_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_int8_cache"
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35HybridPrefixKey = exact_cache_module.Qwen35HybridPrefixKey
Qwen35HybridPrefixSnapshotCache = (
    exact_cache_module.Qwen35HybridPrefixSnapshotCache
)
QWEN35_RECURRENT_INT8_CODEC = codec_module.QWEN35_RECURRENT_INT8_CODEC
Qwen35EncodedRecurrentInt8 = codec_module.Qwen35EncodedRecurrentInt8
QWEN35_HYBRID_PREFIX_EXACT = (
    representation_module.QWEN35_HYBRID_PREFIX_EXACT
)
QWEN35_HYBRID_PREFIX_RECURRENT_INT8 = (
    representation_module.QWEN35_HYBRID_PREFIX_RECURRENT_INT8
)
Qwen35HybridPrefixInt8SnapshotCache = (
    int8_cache_module.Qwen35HybridPrefixInt8SnapshotCache
)

LINEAR_LAYER_COUNT = 18
CONVOLUTION_SHAPE = (2, 3)
RECURRENT_SHAPE = (2, 2, 2)


def _layout(layer_count=LINEAR_LAYER_COUNT):
    return HybridStateLayout(tuple(
        component
        for layer_index in range(layer_count)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                CONVOLUTION_SHAPE,
                torch.bfloat16,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                RECURRENT_SHAPE,
                torch.float32,
            ),
        )
    ))


def _fixture(
    *,
    layer_count=LINEAR_LAYER_COUNT,
    max_entries=4,
    max_bytes=1 << 20,
):
    pool = HybridStateTensorPool(
        _layout(layer_count),
        capacity=4,
        device="cpu",
    )
    leases = tuple(
        HybridStateLease(slot_id, 1, 100 + slot_id)
        for slot_id in range(4)
    )
    for lease in leases:
        pool.activate(lease)
    adapters = tuple(
        Qwen35LayerStateAdapter(pool, layer_index)
        for layer_index in range(layer_count)
    )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    for layer_index, adapter in enumerate(adapters):
        for slot_id in range(4):
            convolution = (
                torch.arange(6, dtype=torch.float32).reshape(
                    CONVOLUTION_SHAPE
                )
                + layer_index * 100
                + slot_id * 10
            ).to(torch.bfloat16)
            recurrent = (
                torch.arange(8, dtype=torch.float32).reshape(
                    RECURRENT_SHAPE
                )
                * 0.03125
                + layer_index
                + slot_id * 0.25
                + 0.125
            )
            adapter.convolution[slot_id].copy_(convolution)
            adapter.recurrent[slot_id].copy_(recurrent)
    cache = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=max_entries,
        max_bytes=max_bytes,
    )
    return pool, leases, adapters, transaction, cache


def _key(
    *,
    token_hash=101,
    terminal_block_hash=201,
    model_fingerprint="qwen35-test-model",
    layout_fingerprint="qwen35-test-layout",
):
    return Qwen35HybridPrefixKey(
        token_hash=token_hash,
        token_count=4,
        terminal_block_hash=terminal_block_hash,
        block_size=4,
        model_fingerprint=model_fingerprint,
        layout_fingerprint=layout_fingerprint,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )


def _tokens(base=1):
    return tuple(range(base, base + 4))


def _blocks(block_id=7, generation=3, block_hash=201):
    return ((block_id, generation, block_hash),)


def _entries(cache):
    return tuple(cache._entries.items())


def _snapshots(cache):
    return tuple(snapshot for _, snapshot in _entries(cache))


def _only_snapshot(cache):
    snapshots = _snapshots(cache)
    assert len(snapshots) == 1
    return snapshots[0]


def _physical_bytes(snapshot):
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in snapshot.layers
        for tensor in (
            layer.convolution_state,
            layer.recurrent_values,
            layer.recurrent_scales,
        )
    )


def _tensor_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _state_rows(adapters, leases):
    slot_ids = tuple(lease.slot_id for lease in leases)
    return tuple(
        (
            adapter.convolution[list(slot_ids)].clone(),
            adapter.recurrent[list(slot_ids)].clone(),
        )
        for adapter in adapters
    )


def _assert_state_rows_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_pair, expected_pair in zip(actual, expected):
        torch.testing.assert_close(actual_pair[0], expected_pair[0])
        torch.testing.assert_close(actual_pair[1], expected_pair[1])


def _zero_state_rows(adapters, leases):
    for adapter in adapters:
        for lease in leases:
            adapter.convolution[lease.slot_id].zero_()
            adapter.recurrent[lease.slot_id].zero_()


def _encoded_from_layer(layer):
    payload_bytes = layer.recurrent_values.untyped_storage().nbytes()
    scale_bytes = layer.recurrent_scales.untyped_storage().nbytes()
    return Qwen35EncodedRecurrentInt8(
        codec=layer.codec,
        values=layer.recurrent_values,
        scales=layer.recurrent_scales,
        source_shape=layer.source_shape,
        source_dtype=layer.source_dtype,
        logical_bytes=(
            layer.recurrent_values.numel()
            * torch.tensor([], dtype=torch.float32).element_size()
        ),
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        encoded_bytes=payload_bytes + scale_bytes,
    )


def _cache_state(cache):
    return {
        "observation": cache.observation_snapshot(),
        "entries": tuple(
            (entry_key, id(snapshot))
            for entry_key, snapshot in cache._entries.items()
        ),
        "intern_total_bytes": cache._intern_total_bytes,
        "intern_records": tuple(sorted(
            (
                id(record.tensor),
                record.refcount,
                record.visible_refcount,
                record.storage_bytes,
                record.key,
            )
            for record in cache._intern_records.values()
        )),
    }


def _assert_state_delta(
    before,
    after,
    *,
    observation_deltas=None,
):
    expected = dict(before["observation"])
    for field, delta in (observation_deltas or {}).items():
        expected[field] += delta
    assert after["observation"] == expected, {
        field: (expected[field], after["observation"].get(field))
        for field in expected
        if after["observation"].get(field) != expected[field]
    }
    assert after["entries"] == before["entries"]
    assert after["intern_total_bytes"] == before["intern_total_bytes"]
    assert after["intern_records"] == before["intern_records"]


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_all_18_layers_publish_atomically_and_charge_encoded_storage():
    _, leases, _, _, cache = _fixture()
    assert cache.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True

    snapshot = _only_snapshot(cache)
    assert tuple(
        layer.layer_index for layer in snapshot.layers
    ) == tuple(range(LINEAR_LAYER_COUNT))
    assert all(
        layer.convolution_state.dtype == torch.bfloat16
        for layer in snapshot.layers
    )
    assert all(
        layer.recurrent_values.dtype == torch.int8
        for layer in snapshot.layers
    )
    assert all(
        layer.recurrent_scales.dtype == torch.float32
        for layer in snapshot.layers
    )
    assert all(
        layer.codec == QWEN35_RECURRENT_INT8_CODEC
        for layer in snapshot.layers
    )
    assert all(
        layer.source_shape == RECURRENT_SHAPE
        for layer in snapshot.layers
    )
    assert all(
        not hasattr(layer, "recurrent_state")
        for layer in snapshot.layers
    )

    convolution_bytes = LINEAR_LAYER_COUNT * 6 * 2
    recurrent_logical_bytes = LINEAR_LAYER_COUNT * 8 * 4
    recurrent_payload_bytes = LINEAR_LAYER_COUNT * 8
    scale_bytes = LINEAR_LAYER_COUNT * 4 * 4
    assert snapshot.accounting.full_fidelity_logical_bytes == (
        convolution_bytes + recurrent_logical_bytes
    )
    assert snapshot.accounting.encoded_physical_bytes == (
        convolution_bytes + recurrent_payload_bytes + scale_bytes
    )
    assert snapshot.accounting.codec_metadata_bytes > 0
    assert snapshot.accounting.temporary_encode_workspace_bytes >= 0
    assert snapshot.accounting.temporary_decode_workspace_bytes == 0
    assert _physical_bytes(snapshot) == (
        snapshot.accounting.encoded_physical_bytes
    )

    observed = cache.observation_snapshot()
    assert observed["representation"] == (
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8
    )
    assert observed["codec"] == QWEN35_RECURRENT_INT8_CODEC
    assert observed["current_bytes"] == (
        observed["current_encoded_physical_bytes"]
    )
    assert observed["current_bytes"] == _physical_bytes(snapshot)
    assert observed["current_full_fidelity_logical_bytes"] == (
        snapshot.accounting.full_fidelity_logical_bytes
    )
    assert observed["current_codec_metadata_bytes"] == (
        snapshot.accounting.codec_metadata_bytes
    )


def test_prepare_precommit_finalize_seal_controls_visibility():
    _, leases, _, _, cache = _fixture()
    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    assert prepared is not None
    assert _snapshots(cache) == ()
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 1

    cache.precommit_publication(prepared)
    assert _snapshots(cache) == ()
    precommitted = cache.observation_snapshot()
    assert precommitted["current_precommitted_references"] == 54
    assert precommitted["current_entries"] == 0

    assert cache.finalize_publication(prepared) is True
    assert len(_snapshots(cache)) == 1
    finalized = cache.observation_snapshot()
    assert finalized["current_prepared_publications"] == 1
    assert finalized["current_entries"] == 1

    cache.seal_publication(prepared)
    sealed = cache.observation_snapshot()
    assert sealed["current_prepared_publications"] == 0
    assert sealed["current_precommitted_references"] == 0
    assert sealed["publication_prepares"] == 1
    assert sealed["publication_precommits"] == 1
    assert sealed["publication_commits"] == 1


def test_prepared_only_rollback_restores_exact_pre_prepare_state():
    _, leases, _, _, cache = _fixture()
    before = _cache_state(cache)

    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    assert prepared is not None
    cache.rollback_publication(prepared)

    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"publication_rollbacks": 1},
    )


def test_abort_current_publication_rolls_back_owned_prepared_state():
    _, leases, _, _, cache = _fixture()
    before = _cache_state(cache)

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
    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"publication_rollbacks": 1},
    )
    assert cache.abort_current_publication() is False


def test_precommitted_rollback_restores_exact_pre_prepare_state():
    _, leases, _, _, cache = _fixture()
    before = _cache_state(cache)

    prepared = cache.prepare_publication(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    )
    assert prepared is not None
    cache.precommit_publication(prepared)
    assert cache.observation_snapshot()[
        "current_precommitted_references"
    ] == LINEAR_LAYER_COUNT * 3
    cache.rollback_publication(prepared)

    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"publication_rollbacks": 1},
    )


def test_late_encode_failure_preserves_previous_entry_and_releases_intern_refs():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    previous = _only_snapshot(cache)
    before = cache.observation_snapshot()

    adapters[17].recurrent[0].fill_(float("nan"))
    _expect_error(
        lambda: cache.publish(key, tokens, blocks, leases[0]),
        ValueError,
        "finite",
    )

    assert _only_snapshot(cache) is previous
    observed = cache.observation_snapshot()
    assert observed["current_prepared_publications"] == 0
    assert observed["current_intern_references"] == before[
        "current_intern_references"
    ]
    assert observed["current_interned_tensors"] == before[
        "current_interned_tensors"
    ]
    assert observed["current_bytes"] == before["current_bytes"]
    assert observed["current_entries"] == before["current_entries"]


def test_encode_workspace_peak_survives_later_layer_failure():
    _, leases, _, _, cache = _fixture()
    original_encode = (
        int8_cache_module
        ._encode_recurrent_with_cuda_workspace_accounting
    )
    encode_calls = 0

    def measure_then_fail(recurrent):
        nonlocal encode_calls
        encode_calls += 1
        if encode_calls == 2:
            raise RuntimeError("injected second-layer encode failure")
        encoded, _ = original_encode(recurrent)
        return encoded, 12345

    int8_cache_module._encode_recurrent_with_cuda_workspace_accounting = (
        measure_then_fail
    )
    try:
        _expect_error(
            lambda: cache.publish(
                _key(),
                _tokens(),
                _blocks(),
                leases[0],
            ),
            RuntimeError,
            "second-layer encode failure",
        )
    finally:
        (
            int8_cache_module
            ._encode_recurrent_with_cuda_workspace_accounting
        ) = original_encode

    assert encode_calls == 2
    observed = cache.observation_snapshot()
    assert observed["peak_temporary_encode_workspace_bytes"] == 12345
    assert observed["current_entries"] == 0


def test_exact_byte_equality_interns_all_three_layer_tensors():
    _, leases, _, _, cache = _fixture()
    assert cache.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    first = _only_snapshot(cache)
    first_bytes = cache.observation_snapshot()["current_bytes"]

    assert cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        leases[0],
    ) is True
    first_snapshot, second_snapshot = _snapshots(cache)
    assert first_snapshot is first
    for first_layer, second_layer in zip(
        first_snapshot.layers,
        second_snapshot.layers,
    ):
        assert (
            first_layer.convolution_state
            is second_layer.convolution_state
        )
        assert first_layer.recurrent_values is second_layer.recurrent_values
        assert first_layer.recurrent_scales is second_layer.recurrent_scales

    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 2
    assert observed["current_bytes"] == first_bytes
    assert observed["current_interned_tensors"] == 54
    assert observed["current_intern_references"] == 108
    assert observed["deduplicated_bytes"] == first_bytes


def test_digest_collision_requires_exact_bytes_before_aliasing():
    _, leases, _, _, cache = _fixture()
    original_digest = int8_cache_module._tensor_digest
    int8_cache_module._tensor_digest = lambda *args: "forced-collision"
    try:
        assert cache.publish(
            _key(),
            _tokens(),
            _blocks(),
            leases[0],
        ) is True
        assert cache.publish(
            _key(token_hash=102, terminal_block_hash=202),
            _tokens(10),
            _blocks(8, 1, 202),
            leases[1],
        ) is True
        assert cache.publish(
            _key(token_hash=103, terminal_block_hash=203),
            _tokens(20),
            _blocks(9, 1, 203),
            leases[1],
        ) is True

        first, second, third = _snapshots(cache)
        for first_layer, second_layer, third_layer in zip(
            first.layers,
            second.layers,
            third.layers,
        ):
            for first_tensor, second_tensor, third_tensor in (
                (
                    first_layer.convolution_state,
                    second_layer.convolution_state,
                    third_layer.convolution_state,
                ),
                (
                    first_layer.recurrent_values,
                    second_layer.recurrent_values,
                    third_layer.recurrent_values,
                ),
                (
                    first_layer.recurrent_scales,
                    second_layer.recurrent_scales,
                    third_layer.recurrent_scales,
                ),
            ):
                assert first_tensor.data_ptr() != second_tensor.data_ptr()
                assert second_tensor is third_tensor
        assert cache.observation_snapshot()["intern_collisions"] > 0
    finally:
        int8_cache_module._tensor_digest = original_digest


def test_mixed_codec_publication_is_rejected_atomically():
    _, leases, _, _, cache = _fixture()
    original_encode = (
        int8_cache_module.encode_qwen35_recurrent_int8_per_row
    )

    def mixed_codec_encode(recurrent):
        encoded = original_encode(recurrent)
        if recurrent.flatten()[0].item() >= 17:
            return replace(encoded, codec="foreign-int8-codec")
        return encoded

    int8_cache_module.encode_qwen35_recurrent_int8_per_row = (
        mixed_codec_encode
    )
    try:
        _expect_error(
            lambda: cache.publish(
                _key(),
                _tokens(),
                _blocks(),
                leases[0],
            ),
            ValueError,
            "codec",
        )
    finally:
        int8_cache_module.encode_qwen35_recurrent_int8_per_row = (
            original_encode
        )

    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 0
    assert observed["current_bytes"] == 0
    assert observed["current_prepared_publications"] == 0
    assert observed["current_interned_tensors"] == 0


def test_forbidden_negative_128_payload_is_rejected_atomically():
    _, leases, _, _, cache = _fixture()
    before = _cache_state(cache)
    original_encode = (
        int8_cache_module.encode_qwen35_recurrent_int8_per_row
    )
    encode_calls = 0

    def encode_with_forbidden_payload(recurrent):
        nonlocal encode_calls
        encode_calls += 1
        encoded = original_encode(recurrent)
        if encode_calls == LINEAR_LAYER_COUNT:
            values = encoded.values.clone()
            values.view(-1)[0] = -128
            return replace(encoded, values=values)
        return encoded

    int8_cache_module.encode_qwen35_recurrent_int8_per_row = (
        encode_with_forbidden_payload
    )
    try:
        _expect_error(
            lambda: cache.publish(
                _key(),
                _tokens(),
                _blocks(),
                leases[0],
            ),
            ValueError,
            "forbidden -128",
        )
    finally:
        int8_cache_module.encode_qwen35_recurrent_int8_per_row = (
            original_encode
        )

    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"validation_failures": 1},
    )


def test_convolution_snapshot_is_exact_detached_contiguous_owned_bf16():
    _, leases, adapters, _, cache = _fixture()
    sources = tuple(
        adapter.convolution[leases[0].slot_id]
        for adapter in adapters
    )
    for source in sources:
        source.requires_grad_(True)
    expected_bytes = tuple(_tensor_bytes(source) for source in sources)

    assert cache.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    snapshot = _only_snapshot(cache)

    for source, expected, layer in zip(
        sources,
        expected_bytes,
        snapshot.layers,
    ):
        convolution = layer.convolution_state
        assert convolution.dtype == torch.bfloat16
        assert convolution.is_contiguous()
        assert convolution.requires_grad is False
        assert convolution.grad_fn is None
        assert _tensor_bytes(convolution) == expected
        assert convolution.data_ptr() != source.data_ptr()
        assert (
            convolution.untyped_storage().data_ptr()
            != source.untyped_storage().data_ptr()
        )


def test_partial_layer_transaction_is_rejected_without_publication():
    _, leases, _, _, cache = _fixture(layer_count=17)
    _expect_error(
        lambda: cache.publish(
            _key(),
            _tokens(),
            _blocks(),
            leases[0],
        ),
        ValueError,
        "exactly 18 layers",
    )
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 0
    assert observed["current_prepared_publications"] == 0


def test_duplicate_layer_identities_are_rejected_without_publication():
    _, leases, adapters, transaction, cache = _fixture()
    transaction.adapters = (
        *adapters[:-1],
        adapters[-2],
    )

    _expect_error(
        lambda: cache.publish(
            _key(),
            _tokens(),
            _blocks(),
            leases[0],
        ),
        ValueError,
        "layer identities are not unique",
    )
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 0
    assert observed["current_prepared_publications"] == 0


def test_unordered_layer_identities_are_rejected_without_publication():
    _, leases, adapters, _, _ = _fixture()
    transaction = Qwen35CrossLayerStateTransaction((
        adapters[1],
        adapters[0],
        *adapters[2:],
    ))
    cache = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=1 << 20,
    )

    _expect_error(
        lambda: cache.publish(
            _key(),
            _tokens(),
            _blocks(),
            leases[0],
        ),
        ValueError,
        "layers are not ordered",
    )
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 0
    assert observed["current_prepared_publications"] == 0


def test_oversize_publication_and_replacement_preserve_resident_entry():
    _, leases, adapters, transaction, _ = _fixture()
    probe = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=1 << 20,
    )
    assert probe.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    resident_bytes = probe.observation_snapshot()["current_bytes"]

    cache = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=resident_bytes,
    )
    key = _key()
    assert cache.publish(
        key,
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    previous = _only_snapshot(cache)
    for adapter in adapters:
        adapter.convolution[1].add_(100)
        adapter.recurrent[1].add_(0.5)

    assert cache.publish(
        key,
        _tokens(),
        _blocks(),
        leases[1],
    ) is False
    assert _only_snapshot(cache) is previous
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 1
    assert observed["current_bytes"] == resident_bytes
    assert observed["replacements"] == 0
    assert observed["oversize_rejections"] == 1


def test_fresh_cache_rejects_single_candidate_larger_than_byte_limit():
    _, leases, _, transaction, probe = _fixture()
    assert probe.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    resident_bytes = probe.observation_snapshot()["current_bytes"]
    cache = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=resident_bytes - 1,
    )
    before = _cache_state(cache)

    assert cache.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is False

    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"oversize_rejections": 1},
    )


def test_finalized_replacement_rollback_restores_exact_previous_state():
    _, leases, _, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    previous = _only_snapshot(cache)
    before = _cache_state(cache)

    prepared = cache.prepare_publication(
        key,
        tokens,
        blocks,
        leases[1],
    )
    cache.precommit_publication(prepared)
    assert cache.finalize_publication(prepared) is True
    assert _only_snapshot(cache) is not previous

    cache.rollback_publication(prepared)
    assert _only_snapshot(cache) is previous
    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"publication_rollbacks": 1},
    )


def test_byte_limit_eviction_and_finalized_rollback_restore_exact_state():
    _, leases, _, transaction, probe = _fixture()
    assert probe.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    one_entry_bytes = probe.observation_snapshot()["current_bytes"]
    cache = Qwen35HybridPrefixInt8SnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=one_entry_bytes * 2,
    )
    identities = (
        (_key(), _tokens(), _blocks(), leases[0]),
        (
            _key(token_hash=102, terminal_block_hash=202),
            _tokens(10),
            _blocks(8, 1, 202),
            leases[1],
        ),
    )
    for key, tokens, blocks, lease in identities:
        assert cache.publish(key, tokens, blocks, lease) is True
    before = _cache_state(cache)

    prepared = cache.prepare_publication(
        _key(token_hash=103, terminal_block_hash=203),
        _tokens(20),
        _blocks(9, 1, 203),
        leases[2],
    )
    assert prepared is not None
    cache.precommit_publication(prepared)
    assert cache.finalize_publication(prepared) is True
    finalized = cache.observation_snapshot()
    assert finalized["current_entries"] == 2
    assert finalized["current_bytes"] <= cache.max_bytes
    assert finalized["evictions"] == before["observation"]["evictions"] + 1
    assert finalized["byte_limit_evictions"] == (
        before["observation"]["byte_limit_evictions"] + 1
    )
    assert tuple(snapshot.key for snapshot in _snapshots(cache)) == (
        identities[1][0],
        prepared.key,
    )

    cache.rollback_publication(prepared)
    _assert_state_delta(
        before,
        _cache_state(cache),
        observation_deltas={"publication_rollbacks": 1},
    )


def test_lru_entry_limit_evicts_oldest_snapshot():
    _, leases, _, _, cache = _fixture(max_entries=2)
    identities = (
        (_key(), _tokens(), _blocks()),
        (
            _key(token_hash=102, terminal_block_hash=202),
            _tokens(10),
            _blocks(8, 1, 202),
        ),
        (
            _key(token_hash=103, terminal_block_hash=203),
            _tokens(20),
            _blocks(9, 1, 203),
        ),
    )
    for key, tokens, blocks in identities:
        assert cache.publish(key, tokens, blocks, leases[0]) is True

    snapshots = _snapshots(cache)
    assert tuple(snapshot.key for snapshot in snapshots) == (
        identities[1][0],
        identities[2][0],
    )
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 2
    assert observed["evictions"] == 1
    assert observed["entry_limit_evictions"] == 1


def test_reader_lease_keeps_snapshot_alive_across_concurrent_eviction():
    _, leases, _, _, cache = _fixture(max_entries=1)
    first_key = _key()
    first_tokens = _tokens()
    first_blocks = _blocks()
    assert cache.publish(
        first_key,
        first_tokens,
        first_blocks,
        leases[0],
    ) is True

    reader = cache.acquire_reader(
        first_key,
        first_tokens,
        first_blocks,
    )
    assert reader is not None
    assert cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        leases[1],
    ) is True

    assert reader.snapshot.key == first_key
    reader.release()
    assert cache.observation_snapshot()["current_reader_leases"] == 0


def test_reader_snapshot_and_take_cannot_mutate_resident_interned_tensors():
    _, leases, _, _, cache = _fixture()
    first_identity = (_key(), _tokens(), _blocks())
    second_identity = (
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
    )
    assert cache.publish(*first_identity, leases[0]) is True
    assert cache.publish(*second_identity, leases[0]) is True

    first_resident, second_resident = _snapshots(cache)
    assert all(
        first_tensor is second_tensor
        for first_tensor, second_tensor in zip(
            cache._snapshot_tensors(first_resident),
            cache._snapshot_tensors(second_resident),
        )
    )
    resident_bytes = tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(first_resident)
    )
    alias_bytes = tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(second_resident)
    )

    reader = cache.acquire_reader(*first_identity)
    assert reader is not None
    exposed = reader.snapshot
    for tensor in cache._snapshot_tensors(exposed):
        tensor.view(torch.uint8).fill_(0xA5)
    assert tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(first_resident)
    ) == resident_bytes
    assert tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(second_resident)
    ) == alias_bytes
    reader.release()

    reader = cache.acquire_reader(*first_identity)
    assert reader is not None
    transferred = reader.take()
    for tensor in cache._snapshot_tensors(transferred):
        tensor.view(torch.uint8).fill_(0x5A)
    assert tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(first_resident)
    ) == resident_bytes
    assert tuple(
        _tensor_bytes(tensor)
        for tensor in cache._snapshot_tensors(second_resident)
    ) == alias_bytes
    reader.release()
    assert cache.observation_snapshot()["current_reader_leases"] == 0


def test_acquire_decodes_pinned_resident_without_public_snapshot_clone():
    _, leases, _, _, cache = _fixture()
    identity = (_key(), _tokens(), _blocks())
    assert cache.publish(*identity, leases[0]) is True
    original_private_snapshot = cache._private_snapshot
    private_snapshot_calls = 0

    def fail_private_snapshot(snapshot):
        nonlocal private_snapshot_calls
        private_snapshot_calls += 1
        raise AssertionError(
            "acquire must not materialize the public private snapshot"
        )

    cache._private_snapshot = fail_private_snapshot
    try:
        assert cache.acquire(*identity, (leases[1], leases[2])) is True
    finally:
        cache._private_snapshot = original_private_snapshot

    assert private_snapshot_calls == 0
    assert cache.observation_snapshot()["current_reader_leases"] == 0


def test_late_layer_17_decode_failure_leaves_destination_unchanged_and_quarantines():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    destination_leases = (leases[1], leases[2])
    before = _state_rows(adapters, destination_leases)
    original_decode = (
        int8_cache_module.decode_qwen35_recurrent_int8_per_row
    )
    decode_calls = 0

    def fail_on_last_layer(encoded, *, device=None):
        nonlocal decode_calls
        decode_calls += 1
        if decode_calls == LINEAR_LAYER_COUNT:
            raise ValueError("injected layer-17 decode failure")
        return original_decode(encoded, device=device)

    int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
        fail_on_last_layer
    )
    try:
        assert cache.acquire(
            key,
            tokens,
            blocks,
            destination_leases,
        ) is False
    finally:
        int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
            original_decode
        )

    assert decode_calls == LINEAR_LAYER_COUNT
    _assert_state_rows_equal(
        _state_rows(adapters, destination_leases),
        before,
    )
    assert _entries(cache) == ()
    observed = cache.observation_snapshot()
    assert observed["quarantines"] == 1
    assert observed["decode_failures"] == 1
    assert observed["misses"] == 1
    assert observed["hits"] == 0
    assert observed["current_reader_leases"] == 0


def _assert_malformed_resident_inventory_is_quarantined(
    mutate_snapshot,
    *,
    mutate_transaction=None,
):
    _, leases, adapters, transaction, cache = _fixture()
    identity = (_key(), _tokens(), _blocks())
    assert cache.publish(*identity, leases[0]) is True
    entry_key, snapshot = _entries(cache)[0]
    cache._entries[entry_key] = replace(
        snapshot,
        layers=mutate_snapshot(snapshot.layers),
    )
    if mutate_transaction is not None:
        mutate_transaction(transaction, adapters)

    assert cache.acquire(*identity, (leases[1],)) is False

    observed = cache.observation_snapshot()
    assert observed["misses"] == 1
    assert observed["quarantines"] == 1
    assert observed["decode_failures"] == 1
    assert observed["missing_layer_rejections"] == 1
    assert observed["current_reader_leases"] == 0
    assert _entries(cache) == ()


def test_empty_resident_layer_inventory_is_quarantined_once():
    _assert_malformed_resident_inventory_is_quarantined(
        lambda layers: (),
    )


def test_wrong_count_resident_layer_inventory_is_quarantined_once():
    _assert_malformed_resident_inventory_is_quarantined(
        lambda layers: layers[:-1],
    )


def test_duplicate_resident_layer_inventory_is_quarantined_once():
    _assert_malformed_resident_inventory_is_quarantined(
        lambda layers: (
            *layers[:-1],
            replace(layers[-1], layer_index=layers[-2].layer_index),
        ),
    )


def test_unordered_resident_layer_inventory_is_quarantined_once():
    _assert_malformed_resident_inventory_is_quarantined(
        lambda layers: (
            layers[1],
            layers[0],
            *layers[2:],
        ),
    )


def test_snapshot_adapter_inventory_mismatch_is_quarantined_once():
    def swap_adapters(transaction, adapters):
        transaction.adapters = (
            adapters[1],
            adapters[0],
            *adapters[2:],
        )

    _assert_malformed_resident_inventory_is_quarantined(
        lambda layers: layers,
        mutate_transaction=swap_adapters,
    )


def test_successful_restore_decodes_recurrent_state_to_fp32():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    snapshot = _only_snapshot(cache)
    expected = tuple(
        codec_module.decode_qwen35_recurrent_int8_per_row(
            _encoded_from_layer(layer)
        )
        for layer in snapshot.layers
    )
    destination_leases = (leases[1], leases[2])
    _zero_state_rows(adapters, destination_leases)

    assert cache.acquire(
        key,
        tokens,
        blocks,
        destination_leases,
    ) is True

    for adapter, expected_layer in zip(adapters, expected):
        for lease in destination_leases:
            restored = adapter.recurrent[lease.slot_id]
            assert restored.dtype == torch.float32
            torch.testing.assert_close(restored, expected_layer)
    observed = cache.observation_snapshot()
    assert observed["hits"] == 1
    assert observed["misses"] == 0
    assert observed["current_reader_leases"] == 0


def test_successful_restore_preserves_exact_bf16_convolution_bytes():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    snapshot = _only_snapshot(cache)
    expected_bytes = tuple(
        _tensor_bytes(layer.convolution_state)
        for layer in snapshot.layers
    )
    destination_leases = (leases[1], leases[2])
    _zero_state_rows(adapters, destination_leases)

    assert cache.acquire(
        key,
        tokens,
        blocks,
        destination_leases,
    ) is True

    for adapter, expected in zip(adapters, expected_bytes):
        for lease in destination_leases:
            restored = adapter.convolution[lease.slot_id]
            assert restored.dtype == torch.bfloat16
            assert _tensor_bytes(restored) == expected


def test_restore_commits_once_only_after_all_18_layers_decode():
    _, leases, _, transaction, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    original_decode = (
        int8_cache_module.decode_qwen35_recurrent_int8_per_row
    )
    original_commit = transaction.commit
    events = []

    def recording_decode(encoded, *, device=None):
        events.append(("decode", len(events)))
        return original_decode(encoded, device=device)

    def recording_commit(commit_leases, candidates):
        events.append(("commit", len(events)))
        return original_commit(commit_leases, candidates)

    int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
        recording_decode
    )
    transaction.commit = recording_commit
    try:
        assert cache.acquire(
            key,
            tokens,
            blocks,
            (leases[1], leases[2]),
        ) is True
    finally:
        transaction.commit = original_commit
        int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
            original_decode
        )

    assert [event[0] for event in events] == (
        ["decode"] * LINEAR_LAYER_COUNT + ["commit"]
    )


def test_commit_failure_rolls_back_every_destination_layer():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    destination_leases = (leases[1], leases[2])
    before = _state_rows(adapters, destination_leases)
    original_copy = adapters[17]._copy_component
    copy_calls = 0

    def fail_during_last_layer(destination, source):
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 3:
            raise RuntimeError("injected layer-17 commit failure")
        return original_copy(destination, source)

    adapters[17]._copy_component = fail_during_last_layer
    try:
        _expect_error(
            lambda: cache.acquire(
                key,
                tokens,
                blocks,
                destination_leases,
            ),
            RuntimeError,
            "layer-17 commit failure",
        )
    finally:
        adapters[17]._copy_component = original_copy

    assert copy_calls == 3
    _assert_state_rows_equal(
        _state_rows(adapters, destination_leases),
        before,
    )
    observed = cache.observation_snapshot()
    assert observed["commit_failures"] == 1
    assert observed["rollback_failures"] == 0
    assert observed["hits"] == 0
    assert observed["misses"] == 0
    assert observed["quarantines"] == 0
    assert observed["current_reader_leases"] == 0


def test_rollback_failure_is_accounted_and_propagated():
    _, leases, adapters, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True
    original_copy_component = adapters[17]._copy_component
    original_tensor_copy = torch.Tensor.copy_
    rollback_started = False

    def fail_commit(destination, source):
        nonlocal rollback_started
        rollback_started = True
        raise RuntimeError("injected commit failure before rollback")

    def fail_rollback(destination, source, *args, **kwargs):
        if rollback_started:
            raise RuntimeError("injected rollback failure")
        return original_tensor_copy(destination, source, *args, **kwargs)

    adapters[17]._copy_component = fail_commit
    torch.Tensor.copy_ = fail_rollback
    try:
        _expect_error(
            lambda: cache.acquire(
                key,
                tokens,
                blocks,
                (leases[1],),
            ),
            RuntimeError,
            "rollback failure",
        )
    finally:
        torch.Tensor.copy_ = original_tensor_copy
        adapters[17]._copy_component = original_copy_component

    observed = cache.observation_snapshot()
    assert observed["commit_failures"] == 1
    assert observed["rollback_failures"] == 1
    assert observed["hits"] == 0
    assert observed["misses"] == 0
    assert observed["quarantines"] == 0
    assert observed["current_reader_leases"] == 0


def test_decode_staging_workspace_is_released_and_accounted_on_success_and_failure():
    for fail_decode in (False, True):
        _, leases, _, _, cache = _fixture()
        key = _key()
        tokens = _tokens()
        blocks = _blocks()
        assert cache.publish(key, tokens, blocks, leases[0]) is True
        original_decode = (
            int8_cache_module.decode_qwen35_recurrent_int8_per_row
        )
        decode_calls = 0

        def observed_decode(encoded, *, device=None):
            nonlocal decode_calls
            decode_calls += 1
            decoded = original_decode(encoded, device=device)
            if fail_decode and decode_calls == LINEAR_LAYER_COUNT:
                raise RuntimeError("injected staging decode failure")
            return decoded

        int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
            observed_decode
        )
        try:
            restored = cache.acquire(
                key,
                tokens,
                blocks,
                (leases[1], leases[2]),
            )
        finally:
            int8_cache_module.decode_qwen35_recurrent_int8_per_row = (
                original_decode
            )

        assert restored is (not fail_decode)
        observed = cache.observation_snapshot()
        assert observed["current_temporary_decode_workspace_bytes"] == 0
        assert (
            observed["current_temporary_decode_cuda_allocated_bytes"]
            == 0
        )
        assert (
            observed["current_temporary_decode_cuda_reserved_bytes"]
            == 0
        )
        assert observed["peak_temporary_decode_workspace_bytes"] > 0
        assert (
            observed["peak_temporary_decode_cuda_allocated_bytes"]
            >= 0
        )
        assert (
            observed["peak_temporary_decode_cuda_reserved_bytes"]
            >= observed["peak_temporary_decode_cuda_allocated_bytes"]
        )
        assert observed["current_reader_leases"] == 0


def test_failed_restore_does_not_refresh_lru_recency():
    _, leases, _, transaction, cache = _fixture(max_entries=2)
    identities = (
        (_key(), _tokens(), _blocks()),
        (
            _key(token_hash=102, terminal_block_hash=202),
            _tokens(10),
            _blocks(8, 1, 202),
        ),
        (
            _key(token_hash=103, terminal_block_hash=203),
            _tokens(20),
            _blocks(9, 1, 203),
        ),
    )
    assert cache.publish(*identities[0], leases[0]) is True
    assert cache.publish(*identities[1], leases[0]) is True
    original_commit = transaction.commit

    def failing_commit(commit_leases, candidates):
        raise RuntimeError("injected recency commit failure")

    transaction.commit = failing_commit
    try:
        _expect_error(
            lambda: cache.acquire(
                *identities[0],
                (leases[1],),
            ),
            RuntimeError,
            "recency commit failure",
        )
    finally:
        transaction.commit = original_commit

    assert cache.publish(*identities[2], leases[0]) is True
    assert cache.acquire(*identities[0], (leases[1],)) is False
    assert cache.acquire(*identities[1], (leases[1],)) is True


def test_reader_use_after_release_and_double_ownership_transfer_are_rejected():
    _, leases, _, _, cache = _fixture()
    key = _key()
    tokens = _tokens()
    blocks = _blocks()
    assert cache.publish(key, tokens, blocks, leases[0]) is True

    released_reader = cache.acquire_reader(key, tokens, blocks)
    assert released_reader is not None
    released_reader.release()
    _expect_error(
        lambda: released_reader.snapshot,
        RuntimeError,
        "released",
    )
    released_reader.release()

    transferred_reader = cache.acquire_reader(key, tokens, blocks)
    assert transferred_reader is not None
    snapshot = transferred_reader.take()
    assert snapshot.key == key
    _expect_error(
        transferred_reader.take,
        RuntimeError,
        "ownership",
    )
    transferred_reader.release()
    assert cache.observation_snapshot()["current_reader_leases"] == 0


def test_invalidation_releases_matching_accounting_refs_and_intern_storage():
    _, leases, _, _, cache = _fixture()
    first_blocks = _blocks()
    second_blocks = _blocks(8, 1, 202)
    assert cache.publish(
        _key(),
        _tokens(),
        first_blocks,
        leases[0],
    ) is True
    first = _only_snapshot(cache)
    assert cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        second_blocks,
        leases[0],
    ) is True
    before = cache.observation_snapshot()
    assert before["current_entries"] == 2
    assert before["current_interned_tensors"] == LINEAR_LAYER_COUNT * 3
    assert before["current_intern_references"] == LINEAR_LAYER_COUNT * 6

    assert cache.invalidate_blocks(first_blocks) == 1
    remaining = _only_snapshot(cache)
    assert remaining is not first
    assert remaining.key == _key(token_hash=102, terminal_block_hash=202)
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 1
    assert observed["invalidations"] == 1
    assert observed["current_bytes"] == before["current_bytes"]
    assert observed["current_encoded_logical_bytes"] * 2 == (
        before["current_encoded_logical_bytes"]
    )
    assert observed["current_full_fidelity_logical_bytes"] * 2 == (
        before["current_full_fidelity_logical_bytes"]
    )
    assert observed["current_codec_metadata_bytes"] * 2 == (
        before["current_codec_metadata_bytes"]
    )
    assert observed["current_interned_tensors"] == LINEAR_LAYER_COUNT * 3
    assert observed["current_intern_references"] == LINEAR_LAYER_COUNT * 3
    assert cache._intern_total_bytes == observed["current_bytes"]
    assert all(
        record.refcount == 1 and record.visible_refcount == 1
        for record in cache._intern_records.values()
    )


def test_cuda_workspace_helper_subtracts_persistent_output_and_clamps():
    helper_name = "_encode_recurrent_with_cuda_workspace_accounting"
    assert hasattr(int8_cache_module, helper_name), (
        f"missing required helper {helper_name}"
    )
    helper = getattr(int8_cache_module, helper_name)
    original_values = {
        name: getattr(int8_cache_module, name, None)
        for name in (
            "_cuda_memory_allocated",
            "_cuda_max_memory_allocated",
            "_cuda_reset_peak_memory_stats",
            "_cuda_synchronize",
            "encode_qwen35_recurrent_int8_per_row",
        )
    }
    calls = []
    encoded = Qwen35EncodedRecurrentInt8(
        values=torch.zeros((2, 2, 2), dtype=torch.int8),
        scales=torch.ones((2, 2), dtype=torch.float32),
        source_shape=(2, 2, 2),
        source_dtype=torch.float32,
        codec=QWEN35_RECURRENT_INT8_CODEC,
        logical_bytes=32,
        payload_bytes=8,
        scale_bytes=16,
        encoded_bytes=24,
    )

    class FakeCudaTensor:
        device = torch.device("cuda", 0)

    def install_peak(peak_allocated):
        int8_cache_module._cuda_memory_allocated = (
            lambda device: 1_000
        )
        int8_cache_module._cuda_max_memory_allocated = (
            lambda device: peak_allocated
        )
        int8_cache_module._cuda_reset_peak_memory_stats = (
            lambda device: calls.append(("reset", device))
        )
        int8_cache_module._cuda_synchronize = (
            lambda device: calls.append(("sync", device))
        )
        int8_cache_module.encode_qwen35_recurrent_int8_per_row = (
            lambda recurrent: encoded
        )

    try:
        install_peak(1_900)
        observed_encoded, workspace_bytes = helper(FakeCudaTensor())
        assert observed_encoded is encoded
        persistent_output_bytes = (
            encoded.values.untyped_storage().nbytes()
            + encoded.scales.untyped_storage().nbytes()
        )
        assert workspace_bytes == (
            1_900 - 1_000 - persistent_output_bytes
        )
        assert calls == [
            ("reset", torch.device("cuda", 0)),
            ("sync", torch.device("cuda", 0)),
        ]

        calls.clear()
        install_peak(1_005)
        _, workspace_bytes = helper(FakeCudaTensor())
        assert workspace_bytes == 0
    finally:
        for name, value in original_values.items():
            if value is None:
                if hasattr(int8_cache_module, name):
                    delattr(int8_cache_module, name)
            else:
                setattr(int8_cache_module, name, value)


def test_cuda_decode_workspace_lifecycle_uses_synchronized_measured_samples():
    _, _, _, _, cache = _fixture()
    device = torch.device("cuda", 0)
    original_values = {
        name: getattr(int8_cache_module, name)
        for name in (
            "_cuda_memory_allocated",
            "_cuda_max_memory_allocated",
            "_cuda_memory_reserved",
            "_cuda_reset_peak_memory_stats",
            "_cuda_synchronize",
        )
    }
    events = []
    allocated_samples = iter((1_000, 1_700, 1_000))
    reserved_samples = iter((2_000, 2_900, 2_400))

    int8_cache_module._cuda_synchronize = (
        lambda observed_device: events.append(
            ("sync", observed_device)
        )
    )
    int8_cache_module._cuda_memory_allocated = (
        lambda observed_device: (
            events.append(("allocated", observed_device))
            or next(allocated_samples)
        )
    )
    int8_cache_module._cuda_max_memory_allocated = (
        lambda observed_device: (
            events.append(("max_allocated", observed_device))
            or 1_850
        )
    )
    int8_cache_module._cuda_memory_reserved = (
        lambda observed_device: (
            events.append(("reserved", observed_device))
            or next(reserved_samples)
        )
    )
    int8_cache_module._cuda_reset_peak_memory_stats = (
        lambda observed_device: events.append(
            ("reset_peak", observed_device)
        )
    )
    candidates = [(
        torch.zeros((2, 2), dtype=torch.bfloat16),
        torch.zeros((2, 2, 2), dtype=torch.float32),
    )]

    def release_staging():
        events.append(("release_staging", len(candidates)))
        candidates.clear()

    try:
        cache._begin_decode_workspace_accounting(device)
        cache._record_decode_workspace(candidates, device)
        cache._finish_decode_workspace_accounting(
            device,
            release_staging,
        )
    finally:
        for name, value in original_values.items():
            setattr(int8_cache_module, name, value)

    assert candidates == []
    assert events == [
        ("sync", device),
        ("allocated", device),
        ("reserved", device),
        ("reset_peak", device),
        ("sync", device),
        ("allocated", device),
        ("max_allocated", device),
        ("reserved", device),
        ("release_staging", 1),
        ("sync", device),
        ("allocated", device),
        ("reserved", device),
    ]
    observed = cache.observation_snapshot()
    assert observed["current_temporary_decode_workspace_bytes"] == 0
    assert (
        observed["current_temporary_decode_cuda_allocated_bytes"]
        == 0
    )
    assert (
        observed["current_temporary_decode_cuda_reserved_bytes"]
        == 400
    )
    assert (
        observed["peak_temporary_decode_cuda_allocated_bytes"]
        == 850
    )
    assert (
        observed["peak_temporary_decode_cuda_reserved_bytes"]
        == 900
    )


def test_cuda_begin_helper_failures_release_reader_and_deferred_snapshot():
    helper_names = (
        "_cuda_synchronize",
        "_cuda_memory_allocated",
        "_cuda_memory_reserved",
        "_cuda_reset_peak_memory_stats",
    )
    device = torch.device("cuda", 0)
    for failing_helper in helper_names:
        _, leases, _, _, cache = _fixture()
        identity = (_key(), _tokens(), _blocks())
        assert cache.publish(*identity, leases[0]) is True
        entry_key = cache._entry_key(identity[0], identity[1])
        original_begin = cache._begin_decode_workspace_accounting
        original_values = {
            name: getattr(int8_cache_module, name)
            for name in helper_names
        }
        detached = False

        def begin_with_fake_cuda(observed_device):
            return original_begin(device)

        def helper(name, result=None):
            def invoke(observed_device):
                nonlocal detached
                if name == failing_helper:
                    if not detached:
                        cache._remove_entry(entry_key)
                        detached = True
                    raise RuntimeError(f"injected begin failure: {name}")
                return result

            return invoke

        cache._begin_decode_workspace_accounting = begin_with_fake_cuda
        int8_cache_module._cuda_synchronize = helper(
            "_cuda_synchronize"
        )
        int8_cache_module._cuda_memory_allocated = helper(
            "_cuda_memory_allocated",
            1_000,
        )
        int8_cache_module._cuda_memory_reserved = helper(
            "_cuda_memory_reserved",
            2_000,
        )
        int8_cache_module._cuda_reset_peak_memory_stats = helper(
            "_cuda_reset_peak_memory_stats"
        )
        try:
            _expect_error(
                lambda: cache.acquire(*identity, (leases[1],)),
                RuntimeError,
                f"injected begin failure: {failing_helper}",
            )
        finally:
            cache._begin_decode_workspace_accounting = original_begin
            for name, value in original_values.items():
                setattr(int8_cache_module, name, value)

        observed = cache.observation_snapshot()
        assert detached is True
        assert observed["current_reader_leases"] == 0
        assert observed["deferred_snapshot_releases"] == 1
        assert cache._deferred_snapshots == {}
        assert cache._reader_counts == {}
        assert observed["current_intern_references"] == 0
        assert observed["current_temporary_decode_workspace_bytes"] == 0
        assert (
            observed["current_temporary_decode_cuda_allocated_bytes"]
            == 0
        )
        assert (
            observed["current_temporary_decode_cuda_reserved_bytes"]
            == 0
        )
        assert cache._decode_cuda_allocated_baseline == 0
        assert cache._decode_cuda_reserved_baseline == 0


def _assert_cuda_record_helper_failure_preserves_resident_entry(
    failing_helper,
):
    _, leases, _, _, cache = _fixture()
    first_identity = (_key(), _tokens(), _blocks())
    second_identity = (
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
    )
    assert cache.publish(*first_identity, leases[0]) is True
    assert cache.publish(*second_identity, leases[1]) is True
    before = cache.observation_snapshot()
    resident_order = tuple(cache._entries)
    resident_snapshots = tuple(
        id(snapshot) for snapshot in cache._entries.values()
    )
    original_begin = cache._begin_decode_workspace_accounting
    original_record = cache._record_decode_workspace
    original_finish = cache._finish_decode_workspace_accounting
    original_values = {
        name: getattr(int8_cache_module, name)
        for name in (
            "_cuda_synchronize",
            "_cuda_memory_allocated",
            "_cuda_max_memory_allocated",
            "_cuda_memory_reserved",
            "_cuda_reset_peak_memory_stats",
        )
    }
    device = torch.device("cuda", 0)
    live_phase = False

    def begin_with_fake_cuda(observed_device):
        return original_begin(device)

    def record_with_fake_cuda(candidates, observed_device):
        nonlocal live_phase
        live_phase = True
        try:
            return original_record(candidates, device)
        finally:
            live_phase = False

    def finish_with_fake_cuda(observed_device, release_staging):
        return original_finish(device, release_staging)

    def helper(name, result=None):
        def invoke(observed_device):
            if live_phase and name == failing_helper:
                raise RuntimeError(
                    f"injected live telemetry failure: {name}"
                )
            return result

        return invoke

    cache._begin_decode_workspace_accounting = begin_with_fake_cuda
    cache._record_decode_workspace = record_with_fake_cuda
    cache._finish_decode_workspace_accounting = finish_with_fake_cuda
    int8_cache_module._cuda_synchronize = helper(
        "_cuda_synchronize"
    )
    int8_cache_module._cuda_memory_allocated = helper(
        "_cuda_memory_allocated",
        1_000,
    )
    int8_cache_module._cuda_max_memory_allocated = helper(
        "_cuda_max_memory_allocated",
        1_500,
    )
    int8_cache_module._cuda_memory_reserved = helper(
        "_cuda_memory_reserved",
        2_000,
    )
    int8_cache_module._cuda_reset_peak_memory_stats = helper(
        "_cuda_reset_peak_memory_stats"
    )
    try:
        _expect_error(
            lambda: cache.acquire(
                *first_identity,
                (leases[2],),
            ),
            RuntimeError,
            f"injected live telemetry failure: {failing_helper}",
        )
    finally:
        cache._begin_decode_workspace_accounting = original_begin
        cache._record_decode_workspace = original_record
        cache._finish_decode_workspace_accounting = original_finish
        for name, value in original_values.items():
            setattr(int8_cache_module, name, value)

    observed = cache.observation_snapshot()
    assert tuple(cache._entries) == resident_order
    assert tuple(
        id(snapshot) for snapshot in cache._entries.values()
    ) == resident_snapshots
    for field in (
        "current_entries",
        "current_bytes",
        "current_interned_tensors",
        "current_intern_references",
        "quarantines",
        "decode_failures",
        "misses",
        "hits",
        "deferred_snapshot_releases",
    ):
        assert observed[field] == before[field], (
            field,
            before[field],
            observed[field],
        )
    assert observed["current_reader_leases"] == 0
    assert cache._deferred_snapshots == {}
    assert cache._reader_counts == {}
    assert observed["current_temporary_decode_workspace_bytes"] == 0
    assert (
        observed["current_temporary_decode_cuda_allocated_bytes"]
        == 0
    )
    assert (
        observed["current_temporary_decode_cuda_reserved_bytes"]
        == 0
    )
    assert cache._decode_cuda_allocated_baseline == 0
    assert cache._decode_cuda_reserved_baseline == 0


def test_cuda_record_synchronize_failure_preserves_resident_entry():
    _assert_cuda_record_helper_failure_preserves_resident_entry(
        "_cuda_synchronize"
    )


def test_cuda_record_allocated_failure_preserves_resident_entry():
    _assert_cuda_record_helper_failure_preserves_resident_entry(
        "_cuda_memory_allocated"
    )


def test_cuda_record_max_allocated_failure_preserves_resident_entry():
    _assert_cuda_record_helper_failure_preserves_resident_entry(
        "_cuda_max_memory_allocated"
    )


def test_cuda_record_reserved_failure_preserves_resident_entry():
    _assert_cuda_record_helper_failure_preserves_resident_entry(
        "_cuda_memory_reserved"
    )


def test_cuda_finish_helper_failures_preserve_restore_error_and_cleanup():
    helper_names = (
        "_cuda_synchronize",
        "_cuda_memory_allocated",
        "_cuda_memory_reserved",
    )
    device = torch.device("cuda", 0)
    for failing_helper in helper_names:
        _, leases, _, transaction, cache = _fixture()
        identity = (_key(), _tokens(), _blocks())
        assert cache.publish(*identity, leases[0]) is True
        original_begin = cache._begin_decode_workspace_accounting
        original_finish = cache._finish_decode_workspace_accounting
        original_commit = transaction.commit
        original_values = {
            name: getattr(int8_cache_module, name)
            for name in (
                "_cuda_synchronize",
                "_cuda_memory_allocated",
                "_cuda_memory_reserved",
                "_cuda_reset_peak_memory_stats",
            )
        }
        staging_refs = []
        finish_phase = False
        detached = False
        entry_key = cache._entry_key(identity[0], identity[1])

        def begin_with_fake_cuda(observed_device):
            return original_begin(device)

        def finish_with_fake_cuda(observed_device, release_staging):
            nonlocal finish_phase
            nonlocal detached
            finish_phase = True
            cache._remove_entry(entry_key)
            detached = True
            return original_finish(device, release_staging)

        def fail_commit(commit_leases, candidates):
            staging_refs.extend(
                weakref.ref(tensor)
                for candidate in candidates
                for tensor in candidate
            )
            raise RuntimeError("injected primary restore failure")

        def helper(name, result=None):
            def invoke(observed_device):
                if finish_phase and name == failing_helper:
                    raise RuntimeError(f"injected finish failure: {name}")
                return result

            return invoke

        cache._begin_decode_workspace_accounting = begin_with_fake_cuda
        cache._finish_decode_workspace_accounting = finish_with_fake_cuda
        transaction.commit = fail_commit
        int8_cache_module._cuda_synchronize = helper(
            "_cuda_synchronize"
        )
        int8_cache_module._cuda_memory_allocated = helper(
            "_cuda_memory_allocated",
            1_000,
        )
        int8_cache_module._cuda_memory_reserved = helper(
            "_cuda_memory_reserved",
            2_000,
        )
        int8_cache_module._cuda_reset_peak_memory_stats = helper(
            "_cuda_reset_peak_memory_stats"
        )
        try:
            _expect_error(
                lambda: cache.acquire(*identity, (leases[1],)),
                RuntimeError,
                "injected primary restore failure",
            )
        finally:
            cache._begin_decode_workspace_accounting = original_begin
            cache._finish_decode_workspace_accounting = original_finish
            transaction.commit = original_commit
            for name, value in original_values.items():
                setattr(int8_cache_module, name, value)

        gc.collect()
        observed = cache.observation_snapshot()
        assert detached is True
        assert staging_refs
        assert all(reference() is None for reference in staging_refs)
        assert observed["current_reader_leases"] == 0
        assert observed["deferred_snapshot_releases"] == 1
        assert cache._deferred_snapshots == {}
        assert cache._reader_counts == {}
        assert observed["current_intern_references"] == 0
        assert observed["current_temporary_decode_workspace_bytes"] == 0
        assert (
            observed["current_temporary_decode_cuda_allocated_bytes"]
            == 0
        )
        assert (
            observed["current_temporary_decode_cuda_reserved_bytes"]
            == 0
        )
        assert cache._decode_cuda_allocated_baseline == 0
        assert cache._decode_cuda_reserved_baseline == 0


def test_clear_releases_all_entries_and_interned_storage():
    _, leases, _, _, cache = _fixture()
    assert cache.publish(
        _key(),
        _tokens(),
        _blocks(),
        leases[0],
    ) is True
    assert cache.publish(
        _key(token_hash=102, terminal_block_hash=202),
        _tokens(10),
        _blocks(8, 1, 202),
        leases[1],
    ) is True

    assert cache.clear() == 2
    observed = cache.observation_snapshot()
    assert observed["current_entries"] == 0
    assert observed["current_bytes"] == 0
    assert observed["current_encoded_physical_bytes"] == 0
    assert observed["current_full_fidelity_logical_bytes"] == 0
    assert observed["current_codec_metadata_bytes"] == 0
    assert observed["current_interned_tensors"] == 0
    assert observed["current_intern_references"] == 0
    assert observed["clears"] == 1


def test_exact_and_p2_caches_keep_distinct_prefix_representations():
    _, leases, _, transaction, int8_cache = _fixture()
    exact_cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=1 << 20,
    )
    key = _key()
    tokens = _tokens()
    blocks = _blocks()

    assert exact_cache.publish(key, tokens, blocks, leases[0]) is True
    assert int8_cache.publish(key, tokens, blocks, leases[0]) is True

    exact_snapshot = tuple(exact_cache._entries.values())[0]
    int8_snapshot = _only_snapshot(int8_cache)
    assert exact_snapshot.key == int8_snapshot.key
    assert hasattr(exact_snapshot, "recurrent_states")
    assert not hasattr(int8_snapshot, "recurrent_states")
    assert int8_cache.observation_snapshot()["representation"] == (
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8
    )
    assert QWEN35_HYBRID_PREFIX_EXACT != (
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8
    )


def main():
    test_all_18_layers_publish_atomically_and_charge_encoded_storage()
    test_prepare_precommit_finalize_seal_controls_visibility()
    test_prepared_only_rollback_restores_exact_pre_prepare_state()
    test_abort_current_publication_rolls_back_owned_prepared_state()
    test_precommitted_rollback_restores_exact_pre_prepare_state()
    test_late_encode_failure_preserves_previous_entry_and_releases_intern_refs()
    test_encode_workspace_peak_survives_later_layer_failure()
    test_exact_byte_equality_interns_all_three_layer_tensors()
    test_digest_collision_requires_exact_bytes_before_aliasing()
    test_mixed_codec_publication_is_rejected_atomically()
    test_forbidden_negative_128_payload_is_rejected_atomically()
    test_convolution_snapshot_is_exact_detached_contiguous_owned_bf16()
    test_partial_layer_transaction_is_rejected_without_publication()
    test_duplicate_layer_identities_are_rejected_without_publication()
    test_unordered_layer_identities_are_rejected_without_publication()
    test_oversize_publication_and_replacement_preserve_resident_entry()
    test_fresh_cache_rejects_single_candidate_larger_than_byte_limit()
    test_finalized_replacement_rollback_restores_exact_previous_state()
    test_byte_limit_eviction_and_finalized_rollback_restore_exact_state()
    test_lru_entry_limit_evicts_oldest_snapshot()
    test_reader_lease_keeps_snapshot_alive_across_concurrent_eviction()
    test_reader_snapshot_and_take_cannot_mutate_resident_interned_tensors()
    test_acquire_decodes_pinned_resident_without_public_snapshot_clone()
    test_late_layer_17_decode_failure_leaves_destination_unchanged_and_quarantines()
    test_empty_resident_layer_inventory_is_quarantined_once()
    test_wrong_count_resident_layer_inventory_is_quarantined_once()
    test_duplicate_resident_layer_inventory_is_quarantined_once()
    test_unordered_resident_layer_inventory_is_quarantined_once()
    test_snapshot_adapter_inventory_mismatch_is_quarantined_once()
    test_successful_restore_decodes_recurrent_state_to_fp32()
    test_successful_restore_preserves_exact_bf16_convolution_bytes()
    test_restore_commits_once_only_after_all_18_layers_decode()
    test_commit_failure_rolls_back_every_destination_layer()
    test_rollback_failure_is_accounted_and_propagated()
    test_decode_staging_workspace_is_released_and_accounted_on_success_and_failure()
    test_failed_restore_does_not_refresh_lru_recency()
    test_reader_use_after_release_and_double_ownership_transfer_are_rejected()
    test_invalidation_releases_matching_accounting_refs_and_intern_storage()
    test_cuda_workspace_helper_subtracts_persistent_output_and_clamps()
    test_cuda_decode_workspace_lifecycle_uses_synchronized_measured_samples()
    test_cuda_begin_helper_failures_release_reader_and_deferred_snapshot()
    test_cuda_record_synchronize_failure_preserves_resident_entry()
    test_cuda_record_allocated_failure_preserves_resident_entry()
    test_cuda_record_max_allocated_failure_preserves_resident_entry()
    test_cuda_record_reserved_failure_preserves_resident_entry()
    test_cuda_finish_helper_failures_preserve_restore_error_and_cleanup()
    test_clear_releases_all_entries_and_interned_storage()
    test_exact_and_p2_caches_keep_distinct_prefix_representations()
    print("qwen35 hybrid prefix INT8 cache tests passed")


if __name__ == "__main__":
    main()
