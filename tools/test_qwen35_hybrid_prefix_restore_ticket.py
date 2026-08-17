import hashlib
import importlib.util
from pathlib import Path
import pickle
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
ticket_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_restore_ticket.py",
)

SamplingParams = sampling_module.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_module.BlockManager
HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateLease = hybrid_module.HybridStateLease
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
Qwen35HybridPrefixRestorePayload = (
    ticket_module.Qwen35HybridPrefixRestorePayload
)
Qwen35HybridPrefixPrepareAck = (
    ticket_module.Qwen35HybridPrefixPrepareAck
)
Qwen35HybridPrefixRestoreParticipant = (
    ticket_module.Qwen35HybridPrefixRestoreParticipant
)
Qwen35HybridPrefixRestoreCoordinator = (
    ticket_module.Qwen35HybridPrefixRestoreCoordinator
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


def _key(block_manager, tokens, participant_count):
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
        model_fingerprint="qwen35-ticket-test-model",
        layout_fingerprint="qwen35-ticket-test-layout",
        tensor_parallel_size=participant_count,
        dtype=torch.float32,
    )


def _state_rows(adapters, slot_id):
    return tuple(
        (
            adapter.convolution[slot_id].clone(),
            adapter.recurrent[slot_id].clone(),
        )
        for adapter in adapters
    )


def _assert_state_equal(actual, expected):
    for actual_pair, expected_pair in zip(actual, expected):
        torch.testing.assert_close(actual_pair[0], expected_pair[0])
        torch.testing.assert_close(actual_pair[1], expected_pair[1])


def _assert_pristine(sequence):
    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert sequence.num_computed_tokens == 0
    assert sequence.hybrid_state_slot_id == -1
    assert sequence.hybrid_state_generation == 0


def _resource_snapshot(fixture):
    return {
        "free_block_count": len(
            fixture["block_manager"].free_block_ids
        ),
        "used_blocks": tuple(sorted(
            fixture["block_manager"].used_block_ids
        )),
        "ref_counts": tuple(
            block.ref_count
            for block in fixture["block_manager"].blocks
        ),
        "allocator_used_slots": fixture[
            "allocator"
        ].observation_snapshot()["used_slots"],
        "allocator_owners": fixture[
            "allocator"
        ].observation_snapshot()["owners"],
        "bindings": tuple(
            tuple(sorted(participant.pool._bindings.items()))
            for participant in fixture["participants"]
        ),
    }


def _fixture(*, participant_count=2, prefix_tokens=(1, 2, 3, 4)):
    block_manager = BlockManager(num_blocks=12, block_size=4)
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
    reservation = block_manager.reserve_exact_prefix(
        tuple(prefix_tokens),
    )
    assert reservation is not None
    block_identities = reservation.block_identities
    block_manager.release_prefix_reservation(reservation)

    allocator = HybridStateSlotAllocator(capacity=4)
    source_lease = allocator.allocate(source_sequence.seq_id)
    key = _key(
        block_manager,
        tuple(prefix_tokens),
        participant_count,
    )
    participants = []
    expected_states = []
    for participant_id in range(participant_count):
        pool = HybridStateTensorPool(
            _layout(),
            capacity=allocator.capacity,
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
        pool.activate(source_lease)
        for layer_offset, adapter in enumerate(adapters):
            base = (
                10000 * (participant_id + 1)
                + 1000 * (layer_offset + 1)
            )
            adapter.convolution[source_lease.slot_id].copy_(
                torch.arange(6).reshape(2, 3) + base
            )
            adapter.recurrent[source_lease.slot_id].copy_(
                torch.arange(8).reshape(2, 2, 2) + base + 500
            )
        expected_states.append(
            _state_rows(adapters, source_lease.slot_id)
        )
        assert cache.publish(
            key,
            tuple(prefix_tokens),
            block_identities,
            source_lease,
        )
        pool.release(source_lease)
        participants.append(
            Qwen35HybridPrefixRestoreParticipant(
                participant_id,
                pool,
                cache,
            )
        )
    allocator.release(source_lease)
    block_manager.deallocate(source_sequence)
    coordinator = Qwen35HybridPrefixRestoreCoordinator(
        block_manager,
        allocator,
        tuple(participants),
    )
    return {
        "block_manager": block_manager,
        "allocator": allocator,
        "participants": tuple(participants),
        "expected_states": tuple(expected_states),
        "key": key,
        "tokens": tuple(prefix_tokens),
        "block_identities": block_identities,
        "coordinator": coordinator,
    }


def _manual_payload(fixture, request_id=900, ticket_id=7):
    lease = fixture["allocator"].allocate(request_id)
    return Qwen35HybridPrefixRestorePayload(
        ticket_id=ticket_id,
        request_id=request_id,
        key=fixture["key"],
        token_ids=fixture["tokens"],
        block_identities=fixture["block_identities"],
        lease=lease,
    )


def test_payload_and_ack_are_pickle_safe_and_validated():
    fixture = _fixture(participant_count=1)
    payload = _manual_payload(fixture)
    ack = Qwen35HybridPrefixPrepareAck(
        ticket_id=payload.ticket_id,
        participant_id=0,
        status="prepared",
    )

    assert pickle.loads(pickle.dumps(payload)) == payload
    assert pickle.loads(pickle.dumps(ack)) == ack

    try:
        Qwen35HybridPrefixPrepareAck(
            ticket_id=1,
            participant_id=0,
            status="unknown",
        )
    except ValueError as error:
        assert "status" in str(error)
    else:
        raise AssertionError("invalid acknowledgement status was accepted")

    fixture["allocator"].release(payload.lease)


def test_participant_prepare_restores_and_duplicate_is_idempotent():
    fixture = _fixture(participant_count=1)
    participant = fixture["participants"][0]
    payload = _manual_payload(fixture)

    first = participant.prepare(payload)
    second = participant.prepare(payload)

    assert first.status == "prepared"
    assert second == first
    restored = _state_rows(
        participant.snapshot_cache.state_transaction.adapters,
        payload.lease.slot_id,
    )
    _assert_state_equal(restored, fixture["expected_states"][0])
    participant.validate_prepared(payload)
    participant.commit(payload)
    assert participant.pool.validate(payload.lease) == payload.lease.slot_id
    fixture["allocator"].release(payload.lease)


def test_participant_miss_and_error_release_pool_binding():
    fixture = _fixture(participant_count=1)
    participant = fixture["participants"][0]
    participant.snapshot_cache.clear()
    payload = _manual_payload(fixture, ticket_id=11)

    miss = participant.prepare(payload)

    assert miss.status == "miss"
    assert participant.pool._bindings == {}
    fixture["allocator"].release(payload.lease)

    fixture = _fixture(participant_count=1)
    participant = fixture["participants"][0]
    payload = _manual_payload(fixture, ticket_id=12)
    original_acquire = participant.snapshot_cache.acquire

    def failing_acquire(*args, **kwargs):
        raise RuntimeError("injected restore failure")

    participant.snapshot_cache.acquire = failing_acquire
    try:
        error = participant.prepare(payload)
    finally:
        participant.snapshot_cache.acquire = original_acquire

    assert error.status == "error"
    assert "injected restore failure" in error.detail
    assert participant.pool._bindings == {}
    fixture["allocator"].release(payload.lease)


def test_participant_prepare_cleanup_failure_remains_rollbackable():
    fixture = _fixture(participant_count=1)
    participant = fixture["participants"][0]
    payload = _manual_payload(fixture, ticket_id=15)
    original_acquire = participant.snapshot_cache.acquire
    original_release = participant.pool.release
    release_calls = []

    def failing_acquire(*args, **kwargs):
        raise RuntimeError("restore failed before cleanup")

    def fail_first_release(lease):
        release_calls.append(lease)
        if len(release_calls) == 1:
            raise RuntimeError("local cleanup failed")
        return original_release(lease)

    participant.snapshot_cache.acquire = failing_acquire
    participant.pool.release = fail_first_release
    try:
        acknowledgement = participant.prepare(payload)
        assert acknowledgement.status == "error"
        assert "local cleanup failed" in acknowledgement.detail
        assert participant.pool._bindings
        participant.rollback(payload)
    finally:
        participant.snapshot_cache.acquire = original_acquire
        participant.pool.release = original_release

    assert len(release_calls) == 2
    assert participant.pool._bindings == {}
    participant.rollback(payload)
    fixture["allocator"].release(payload.lease)


def test_participant_conflict_rollback_and_terminal_transitions_fail_closed():
    fixture = _fixture(participant_count=1)
    participant = fixture["participants"][0]
    payload = _manual_payload(fixture, ticket_id=13)
    assert participant.prepare(payload).status == "prepared"
    conflicting = Qwen35HybridPrefixRestorePayload(
        ticket_id=payload.ticket_id,
        request_id=payload.request_id + 1,
        key=payload.key,
        token_ids=payload.token_ids,
        block_identities=payload.block_identities,
        lease=HybridStateLease(
            slot_id=payload.lease.slot_id,
            generation=payload.lease.generation,
            request_id=payload.request_id + 1,
        ),
    )

    conflict_ack = participant.prepare(conflicting)

    assert conflict_ack.status == "error"
    participant.rollback(payload)
    assert participant.pool._bindings == {}
    participant.rollback(payload)
    assert participant.prepare(payload).status == "error"

    committed = _manual_payload(fixture, request_id=901, ticket_id=14)
    assert participant.prepare(committed).status == "prepared"
    participant.commit(committed)
    try:
        participant.rollback(committed)
    except RuntimeError as error:
        assert "committed" in str(error)
    else:
        raise AssertionError("committed participant rollback was accepted")
    fixture["allocator"].release(committed.lease)
    fixture["allocator"].release(payload.lease)


def test_reserve_holds_complete_table_and_lease_without_publication():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9, 10])
    before = _resource_snapshot(fixture)

    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )

    assert ticket is not None
    assert ticket.state == "reserved"
    assert ticket.reservation.prefix_block_count == 1
    assert ticket.reservation.new_block_count == 1
    assert len(ticket.reservation.block_ids) == destination.num_blocks
    assert fixture["allocator"].lease_for_request(
        destination.seq_id
    ) == ticket.payload.lease
    _assert_pristine(destination)
    after = _resource_snapshot(fixture)
    assert len(after["used_blocks"]) == len(before["used_blocks"]) + 2
    fixture["coordinator"].rollback(ticket)
    _assert_pristine(destination)
    assert _resource_snapshot(fixture) == before


def test_shorter_kv_prefix_is_clean_miss_with_no_owned_resources():
    fixture = _fixture(prefix_tokens=(1, 2, 3, 4, 5, 6, 7, 8))
    cached_block = fixture["block_manager"].blocks[
        fixture["block_identities"][1][0]
    ]
    fixture["block_manager"]._unregister_cached_block(
        cached_block.block_id
    )
    cached_block.hash = -1
    cached_block.token_ids = []
    destination = _sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
    before_free_count = len(fixture["block_manager"].free_block_ids)

    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )

    assert ticket is None
    _assert_pristine(destination)
    assert len(
        fixture["block_manager"].free_block_ids
    ) == before_free_count
    assert fixture["block_manager"].used_block_ids == set()
    assert fixture["allocator"].observation_snapshot()["used_slots"] == 0


def test_zero_kv_prefix_is_clean_miss_with_no_owned_resources():
    fixture = _fixture()
    fixture["block_manager"].clear_reusable_cache()
    destination = _sequence([1, 2, 3, 4, 9])
    before = _resource_snapshot(fixture)

    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )

    assert ticket is None
    _assert_pristine(destination)
    assert _resource_snapshot(fixture) == before


def test_reserve_rejects_identity_tp_and_dirty_request_before_publication():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    before = _resource_snapshot(fixture)

    wrong_tokens = (1, 2, 3, 5)
    try:
        fixture["coordinator"].reserve(
            destination,
            fixture["key"],
            wrong_tokens,
        )
    except ValueError as error:
        assert "token" in str(error)
    else:
        raise AssertionError("wrong exact tokens were accepted")

    wrong_tp_key = Qwen35HybridPrefixKey(
        **{
            **fixture["key"].__dict__,
            "tensor_parallel_size": 1,
        }
    )
    try:
        fixture["coordinator"].reserve(
            destination,
            wrong_tp_key,
            fixture["tokens"],
        )
    except ValueError as error:
        assert "participant" in str(error)
    else:
        raise AssertionError("wrong participant count was accepted")

    destination.hybrid_state_slot_id = 0
    try:
        fixture["coordinator"].reserve(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except ValueError as error:
        assert "hybrid state" in str(error)
    else:
        raise AssertionError("dirty destination was accepted")
    destination.hybrid_state_slot_id = -1
    assert _resource_snapshot(fixture) == before


def test_reserve_accepts_independent_token_hash_identity():
    fixture = _fixture(participant_count=1)
    destination = _sequence([1, 2, 3, 4, 9])
    independent_key = Qwen35HybridPrefixKey(
        **{
            **fixture["key"].__dict__,
            "token_hash": fixture["key"].token_hash + 1,
        }
    )

    ticket = fixture["coordinator"].reserve(
        destination,
        independent_key,
        fixture["tokens"],
    )

    assert ticket is not None
    assert ticket.payload.key == independent_key
    fixture["coordinator"].rollback(ticket)
    _assert_pristine(destination)


def test_reserve_allocator_exhaustion_releases_complete_kv_reservation():
    fixture = _fixture()
    blockers = tuple(
        fixture["allocator"].allocate(1000 + index)
        for index in range(fixture["allocator"].capacity)
    )
    destination = _sequence([1, 2, 3, 4, 9])
    before = _resource_snapshot(fixture)

    try:
        fixture["coordinator"].reserve(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "slots exhausted" in str(error)
    else:
        raise AssertionError("allocator exhaustion was swallowed")

    _assert_pristine(destination)
    assert _resource_snapshot(fixture) == before
    for blocker in blockers:
        fixture["allocator"].release(blocker)


def test_prepare_all_participants_then_commit_publishes_once():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9, 10])
    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )
    assert ticket is not None

    acknowledgements = fixture["coordinator"].prepare(ticket)

    assert tuple(ack.status for ack in acknowledgements) == (
        "prepared",
        "prepared",
    )
    assert ticket.state == "prepared"
    try:
        fixture["coordinator"].prepare(ticket)
    except RuntimeError as error:
        assert "preparable" in str(error)
    else:
        raise AssertionError("repeated coordinator prepare was accepted")
    _assert_pristine(destination)
    for participant, expected in zip(
        fixture["participants"],
        fixture["expected_states"],
    ):
        restored = _state_rows(
            participant.snapshot_cache.state_transaction.adapters,
            ticket.payload.lease.slot_id,
        )
        _assert_state_equal(restored, expected)

    fixture["coordinator"].commit(ticket)

    assert ticket.state == "committed"
    assert destination.block_table == list(ticket.reservation.block_ids)
    assert destination.num_cached_tokens == len(fixture["tokens"])
    assert destination.num_computed_tokens == len(fixture["tokens"])
    assert destination.hybrid_state_slot_id == ticket.payload.lease.slot_id
    assert (
        destination.hybrid_state_generation
        == ticket.payload.lease.generation
    )
    for participant in fixture["participants"]:
        assert participant.pool.validate(
            ticket.payload.lease
        ) == ticket.payload.lease.slot_id
    try:
        fixture["coordinator"].commit(ticket)
    except RuntimeError as error:
        assert "committable" in str(error)
    else:
        raise AssertionError("repeated coordinator commit was accepted")


def test_participant_miss_rolls_back_earlier_rank_and_engine_resources():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    before = _resource_snapshot(fixture)
    fixture["participants"][1].snapshot_cache.clear()
    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )
    assert ticket is not None

    acknowledgements = fixture["coordinator"].prepare(ticket)

    assert tuple(ack.status for ack in acknowledgements) == (
        "prepared",
        "miss",
    )
    assert ticket.state == "rolled_back"
    _assert_pristine(destination)
    assert _resource_snapshot(fixture) == before


def test_participant_error_rolls_back_every_resource():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    before = _resource_snapshot(fixture)
    failing_participant = fixture["participants"][1]
    original_acquire = failing_participant.snapshot_cache.acquire

    def failing_acquire(*args, **kwargs):
        raise RuntimeError("rank restore failed")

    failing_participant.snapshot_cache.acquire = failing_acquire
    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )
    assert ticket is not None
    try:
        acknowledgements = fixture["coordinator"].prepare(ticket)
    finally:
        failing_participant.snapshot_cache.acquire = original_acquire

    assert tuple(ack.status for ack in acknowledgements) == (
        "prepared",
        "error",
    )
    assert "rank restore failed" in acknowledgements[-1].detail
    assert ticket.state == "rolled_back"
    _assert_pristine(destination)
    assert _resource_snapshot(fixture) == before


def test_prepare_cleanup_failure_is_explicit_and_releases_engine_resources():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    fixture["participants"][1].snapshot_cache.clear()
    first_participant = fixture["participants"][0]
    original_rollback = first_participant.rollback

    def failing_rollback(payload):
        raise RuntimeError("prepare cleanup failed")

    first_participant.rollback = failing_rollback
    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )
    assert ticket is not None
    try:
        fixture["coordinator"].prepare(ticket)
    except RuntimeError as error:
        assert "prepare cleanup failed" in str(error)
    else:
        raise AssertionError("prepare cleanup failure was swallowed")
    finally:
        first_participant.rollback = original_rollback

    assert ticket.state == "rollback_failed"
    _assert_pristine(destination)
    assert fixture["allocator"].lease_for_request(
        destination.seq_id
    ) is None
    assert ticket.reservation.state == "released"


def test_explicit_rollback_from_reserved_and_prepared_is_complete():
    for prepare_first in (False, True):
        fixture = _fixture()
        destination = _sequence([1, 2, 3, 4, 9])
        before = _resource_snapshot(fixture)
        ticket = fixture["coordinator"].reserve(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
        assert ticket is not None
        if prepare_first:
            fixture["coordinator"].prepare(ticket)
            assert ticket.state == "prepared"

        fixture["coordinator"].rollback(ticket)

        assert ticket.state == "rolled_back"
        _assert_pristine(destination)
        assert _resource_snapshot(fixture) == before
        try:
            fixture["coordinator"].rollback(ticket)
        except RuntimeError as error:
            assert "rollbackable" in str(error)
        else:
            raise AssertionError("repeated coordinator rollback was accepted")


def test_participant_rollback_failure_is_not_reported_as_rolled_back():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    ticket = fixture["coordinator"].reserve(
        destination,
        fixture["key"],
        fixture["tokens"],
    )
    assert ticket is not None
    fixture["coordinator"].prepare(ticket)
    failing_participant = fixture["participants"][1]
    original_rollback = failing_participant.rollback

    def failing_rollback(payload):
        raise RuntimeError("injected rollback failure")

    failing_participant.rollback = failing_rollback
    try:
        fixture["coordinator"].rollback(ticket)
    except RuntimeError as error:
        assert "injected rollback failure" in str(error)
    else:
        raise AssertionError("participant rollback failure was swallowed")
    finally:
        failing_participant.rollback = original_rollback

    assert ticket.state == "rollback_failed"
    _assert_pristine(destination)
    assert fixture["allocator"].lease_for_request(
        destination.seq_id
    ) is None
    assert ticket.reservation.state == "released"
    next_destination = _sequence([1, 2, 3, 4, 8])
    try:
        fixture["coordinator"].reserve(
            next_destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "poisoned" in str(error)
    else:
        raise AssertionError(
            "coordinator reused resources after rollback failure"
        )


def test_precommit_stale_allocator_participant_reservation_and_request_fail():
    scenarios = (
        "allocator",
        "participant",
        "reservation",
        "request",
    )
    for scenario in scenarios:
        fixture = _fixture()
        destination = _sequence([1, 2, 3, 4, 9])
        ticket = fixture["coordinator"].reserve(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
        assert ticket is not None
        fixture["coordinator"].prepare(ticket)
        assert ticket.state == "prepared"

        if scenario == "allocator":
            fixture["allocator"].release(ticket.payload.lease)
        elif scenario == "participant":
            ticket.participant_ids = (
                ticket.participant_ids[0] + 100,
                *ticket.participant_ids[1:],
            )
        elif scenario == "reservation":
            block_id = ticket.reservation.block_ids[0]
            fixture["block_manager"].blocks[block_id].generation += 1
        else:
            destination.hybrid_state_generation = 99

        try:
            fixture["coordinator"].commit(ticket)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"stale {scenario} precommit state was accepted"
            )
        assert ticket.state == "prepared"
        assert destination.block_table == []
        assert destination.num_cached_tokens == 0
        if scenario != "request":
            assert destination.hybrid_state_slot_id == -1
            assert destination.hybrid_state_generation == 0
        fixture["coordinator"].rollback(ticket)
        assert ticket.state == "rolled_back"


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 hybrid prefix restore ticket tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
