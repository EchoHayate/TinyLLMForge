import importlib.util
import hashlib
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


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_module = types.ModuleType("xxhash")
xxhash_module.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_module)

sampling_module = _load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_module = _load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
block_module = _load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
hybrid_module = _load_module(
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
acquisition_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_acquisition",
    "tinyvllm/engine/qwen35_hybrid_prefix_acquisition.py",
)

SamplingParams = sampling_module.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_module.BlockManager
HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35HybridPrefixKey = cache_module.Qwen35HybridPrefixKey
Qwen35HybridPrefixSnapshotCache = (
    cache_module.Qwen35HybridPrefixSnapshotCache
)
Qwen35HybridPrefixAcquireCoordinator = (
    acquisition_module.Qwen35HybridPrefixAcquireCoordinator
)
Sequence.block_size = 4


def _layout():
    return HybridStateLayout(tuple(
        component
        for layer_index in (0, 2)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (2, 3),
                torch.float32,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 2, 2),
                torch.float32,
            ),
        )
    ))


def _sequence(tokens):
    return Sequence(
        list(tokens),
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
        ),
    )


def _key(block_manager, tokens):
    prefix_hash = -1
    for start in range(0, len(tokens), block_manager.block_size):
        prefix_hash = block_manager.compute_hash(
            list(tokens[start:start + block_manager.block_size]),
            prefix_hash,
        )
    return Qwen35HybridPrefixKey(
        token_hash=prefix_hash,
        token_count=len(tokens),
        terminal_block_hash=prefix_hash,
        block_size=block_manager.block_size,
        model_fingerprint="qwen35-test-model",
        layout_fingerprint="qwen35-test-layout",
        tensor_parallel_size=1,
        dtype=torch.float32,
    )


def _state_rows(adapters, slot_ids):
    return tuple(
        (
            adapter.convolution[list(slot_ids)].clone(),
            adapter.recurrent[list(slot_ids)].clone(),
        )
        for adapter in adapters
    )


def _assert_pristine(sequence):
    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert sequence.num_computed_tokens == 0
    assert sequence.hybrid_state_slot_id == -1
    assert sequence.hybrid_state_generation == 0


def _fixture(*, capacity=4, prefix_tokens=(1, 2, 3, 4)):
    block_manager = BlockManager(num_blocks=8, block_size=4)
    source_sequence = _sequence(prefix_tokens)
    block_manager.allocate(
        source_sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(
        source_sequence,
        0,
        len(source_sequence),
    )

    allocator = HybridStateSlotAllocator(capacity)
    pool = HybridStateTensorPool(
        _layout(),
        capacity=capacity,
        device="cpu",
    )
    adapters = (
        Qwen35LayerStateAdapter(pool, 0),
        Qwen35LayerStateAdapter(pool, 2),
    )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=8,
        max_bytes=1 << 20,
    )
    source_lease = allocator.allocate(source_sequence.seq_id)
    pool.activate(source_lease)
    for layer_offset, adapter in enumerate(adapters):
        adapter.convolution[source_lease.slot_id].copy_(
            torch.arange(6).reshape(2, 3)
            + 1000 * (layer_offset + 1)
        )
        adapter.recurrent[source_lease.slot_id].copy_(
            torch.arange(8).reshape(2, 2, 2)
            + 2000 * (layer_offset + 1)
        )
    expected_state = _state_rows(
        adapters,
        (source_lease.slot_id,),
    )
    reservation = block_manager.reserve_exact_prefix(
        tuple(prefix_tokens),
    )
    key = _key(block_manager, tuple(prefix_tokens))
    assert cache.publish(
        key,
        tuple(prefix_tokens),
        reservation.block_identities,
        source_lease,
    )
    block_manager.release_prefix_reservation(reservation)
    pool.release(source_lease)
    allocator.release(source_lease)
    block_manager.deallocate(source_sequence)
    coordinator = Qwen35HybridPrefixAcquireCoordinator(
        block_manager,
        allocator,
        pool,
        cache,
    )
    return {
        "block_manager": block_manager,
        "allocator": allocator,
        "pool": pool,
        "adapters": adapters,
        "cache": cache,
        "coordinator": coordinator,
        "key": key,
        "tokens": tuple(prefix_tokens),
        "expected_state": expected_state,
    }


def test_exact_acquire_attaches_kv_lease_and_restores_state():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])

    assert fixture["coordinator"].acquire(
        (destination,),
        fixture["key"],
        fixture["tokens"],
    )

    assert destination.num_cached_tokens == 4
    assert destination.num_computed_tokens == 4
    assert len(destination.block_table) == 1
    lease = fixture["allocator"].lease_for_request(destination.seq_id)
    assert lease is not None
    assert destination.hybrid_state_slot_id == lease.slot_id
    assert destination.hybrid_state_generation == lease.generation
    restored = _state_rows(fixture["adapters"], (lease.slot_id,))
    for actual, expected in zip(restored, fixture["expected_state"]):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


def test_exact_acquire_broadcasts_one_snapshot_to_multiple_requests():
    fixture = _fixture(capacity=3)
    destinations = (
        _sequence([1, 2, 3, 4, 9]),
        _sequence([1, 2, 3, 4, 8]),
        _sequence([1, 2, 3, 4, 7]),
    )

    assert fixture["coordinator"].acquire(
        destinations,
        fixture["key"],
        fixture["tokens"],
    )

    leases = tuple(
        fixture["allocator"].lease_for_request(sequence.seq_id)
        for sequence in destinations
    )
    assert all(lease is not None for lease in leases)
    block_id = destinations[0].block_table[0]
    assert fixture["block_manager"].blocks[block_id].ref_count == 3
    restored = _state_rows(
        fixture["adapters"],
        tuple(lease.slot_id for lease in leases),
    )
    for actual, expected in zip(restored, fixture["expected_state"]):
        torch.testing.assert_close(
            actual[0],
            expected[0].expand(3, *expected[0].shape[1:]),
        )
        torch.testing.assert_close(
            actual[1],
            expected[1].expand(3, *expected[1].shape[1:]),
        )


def test_missing_kv_returns_false_before_state_allocation():
    fixture = _fixture()
    fixture["block_manager"].clear_reusable_cache()
    destination = _sequence([1, 2, 3, 4, 9])

    assert fixture["coordinator"].acquire(
        (destination,),
        fixture["key"],
        fixture["tokens"],
    ) is False

    _assert_pristine(destination)
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0


def test_stale_snapshot_generation_miss_releases_every_resource():
    fixture = _fixture()
    replacement = _sequence(fixture["tokens"])
    fixture["block_manager"].allocate(
        replacement,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    fixture["block_manager"].commit_prefill(
        replacement,
        0,
        len(replacement),
    )
    fixture["block_manager"].deallocate(replacement)
    destination = _sequence([1, 2, 3, 4, 9])

    assert fixture["coordinator"].acquire(
        (destination,),
        fixture["key"],
        fixture["tokens"],
    ) is False

    _assert_pristine(destination)
    assert all(
        block.ref_count == 0
        for block in fixture["block_manager"].blocks
    )
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0
    assert fixture["pool"]._bindings == {}
    assert fixture["cache"].observation_snapshot()[
        "stale_block_misses"
    ] == 1


def test_state_exhaustion_releases_reserved_kv():
    fixture = _fixture(capacity=1)
    blocker = fixture["allocator"].allocate(999)
    fixture["pool"].activate(blocker)
    destination = _sequence([1, 2, 3, 4, 9])

    try:
        fixture["coordinator"].acquire(
            (destination,),
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "slots exhausted" in str(error)
    else:
        raise AssertionError("state exhaustion was swallowed")

    _assert_pristine(destination)
    assert all(
        block.ref_count == 0
        for block in fixture["block_manager"].blocks
    )
    assert fixture["allocator"].lease_for_request(999) == blocker
    assert fixture["pool"]._bindings == {
        blocker.slot_id: (blocker.request_id, blocker.generation),
    }


def test_activation_and_restore_failures_roll_back_all_resources():
    fixture = _fixture(capacity=2)
    destinations = (
        _sequence([1, 2, 3, 4, 9]),
        _sequence([1, 2, 3, 4, 8]),
    )
    original_activate = fixture["pool"].activate
    activate_calls = []

    def failing_activate(lease):
        activate_calls.append(lease)
        if len(activate_calls) == 2:
            raise RuntimeError("injected activation failure")
        return original_activate(lease)

    fixture["pool"].activate = failing_activate
    try:
        fixture["coordinator"].acquire(
            destinations,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert str(error) == "injected activation failure"
    else:
        raise AssertionError("activation failure was swallowed")
    finally:
        fixture["pool"].activate = original_activate
    for sequence in destinations:
        _assert_pristine(sequence)
    assert fixture["pool"]._bindings == {}
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0

    original_acquire = fixture["cache"].acquire

    def failing_restore(*args, **kwargs):
        raise RuntimeError("injected restore failure")

    fixture["cache"].acquire = failing_restore
    try:
        fixture["coordinator"].acquire(
            destinations,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert str(error) == "injected restore failure"
    else:
        raise AssertionError("restore failure was swallowed")
    finally:
        fixture["cache"].acquire = original_acquire
    for sequence in destinations:
        _assert_pristine(sequence)
    assert fixture["pool"]._bindings == {}
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0
    assert all(
        block.ref_count == 0
        for block in fixture["block_manager"].blocks
    )


def test_snapshot_miss_and_validation_fail_before_visible_mutation():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    wrong_key = Qwen35HybridPrefixKey(
        token_hash=fixture["key"].token_hash,
        token_count=fixture["key"].token_count,
        terminal_block_hash=fixture["key"].terminal_block_hash,
        block_size=fixture["key"].block_size,
        model_fingerprint="different-model",
        layout_fingerprint=fixture["key"].layout_fingerprint,
        tensor_parallel_size=fixture["key"].tensor_parallel_size,
        dtype=fixture["key"].dtype,
    )
    assert fixture["coordinator"].acquire(
        (destination,),
        wrong_key,
        fixture["tokens"],
    ) is False
    _assert_pristine(destination)
    assert fixture["pool"]._bindings == {}
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0

    invalid_cases = (
        ((destination, destination), fixture["key"], fixture["tokens"]),
        ((destination,), fixture["key"], fixture["tokens"][:-1]),
        (
            (destination,),
            Qwen35HybridPrefixKey(
                token_hash=fixture["key"].token_hash,
                token_count=fixture["key"].token_count,
                terminal_block_hash=fixture["key"].terminal_block_hash,
                block_size=2,
                model_fingerprint=fixture["key"].model_fingerprint,
                layout_fingerprint=fixture["key"].layout_fingerprint,
                tensor_parallel_size=1,
                dtype=torch.float32,
            ),
            fixture["tokens"],
        ),
        (
            (_sequence([9, 8, 7, 6, 5]),),
            fixture["key"],
            fixture["tokens"],
        ),
        (
            (_sequence([1, 2, 3]),),
            fixture["key"],
            fixture["tokens"],
        ),
    )
    for sequences, key, tokens in invalid_cases:
        try:
            fixture["coordinator"].acquire(sequences, key, tokens)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid acquisition was accepted")
        for sequence in sequences:
            _assert_pristine(sequence)

    destination.block_table.append(3)
    try:
        fixture["coordinator"].acquire(
            (destination,),
            fixture["key"],
            fixture["tokens"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dirty destination was accepted")


def test_successful_transfer_uses_existing_block_deallocation():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    assert fixture["coordinator"].acquire(
        (destination,),
        fixture["key"],
        fixture["tokens"],
    )
    block_id = destination.block_table[0]

    fixture["block_manager"].deallocate(destination)

    assert fixture["block_manager"].blocks[block_id].ref_count == 0
    assert block_id in fixture["block_manager"].free_block_ids


def test_constructor_rejects_pool_mismatch():
    fixture = _fixture()
    other_pool = HybridStateTensorPool(
        _layout(),
        capacity=4,
        device="cpu",
    )
    try:
        Qwen35HybridPrefixAcquireCoordinator(
            fixture["block_manager"],
            fixture["allocator"],
            other_pool,
            fixture["cache"],
        )
    except ValueError as error:
        assert "pool" in str(error)
    else:
        raise AssertionError("mismatched state pool was accepted")


def main():
    test_exact_acquire_attaches_kv_lease_and_restores_state()
    test_exact_acquire_broadcasts_one_snapshot_to_multiple_requests()
    test_missing_kv_returns_false_before_state_allocation()
    test_stale_snapshot_generation_miss_releases_every_resource()
    test_state_exhaustion_releases_reserved_kv()
    test_activation_and_restore_failures_roll_back_all_resources()
    test_snapshot_miss_and_validation_fail_before_visible_mutation()
    test_successful_transfer_uses_existing_block_deallocation()
    test_constructor_rejects_pool_mismatch()
    print("qwen35 hybrid prefix acquisition tests passed")


if __name__ == "__main__":
    main()
