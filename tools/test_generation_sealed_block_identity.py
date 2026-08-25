from dataclasses import replace
import hashlib
import importlib.util
import os
import sys
import types

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        os.path.join(ROOT, package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)


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


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(ROOT, relative_path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sampling = load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_module = load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
block_manager_module = load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
SamplingParams = sampling.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_manager_module.BlockManager
Sequence.block_size = 4


def make_sequence(token_ids):
    return Sequence(
        list(token_ids),
        SamplingParams(
            temperature=0.0,
            max_tokens=32,
        ),
    )


def make_allocated_fixture():
    manager = BlockManager(num_blocks=16, block_size=4)
    sequence = make_sequence([1, 2, 3, 4, 5])
    manager.allocate(sequence, publish_hashes=False)
    return manager, sequence


def test_ownership_generation_advances_for_allocate_and_deallocate():
    manager = BlockManager(num_blocks=16, block_size=4)
    sequence = make_sequence([1, 2, 3, 4, 5])
    initial = manager.ownership_generation

    manager.allocate(sequence, publish_hashes=False)
    allocated = manager.ownership_generation
    manager.deallocate(sequence)

    assert allocated > initial
    assert manager.ownership_generation > allocated


def test_hash_publication_does_not_advance_ownership_generation():
    manager, sequence = make_allocated_fixture()
    before = manager.ownership_generation

    manager.publish_full_blocks(
        sequence,
        materialized_tokens=4,
    )

    assert manager.ownership_generation == before


def test_cached_prefix_reference_change_advances_ownership_generation():
    manager, source = make_allocated_fixture()
    manager.publish_full_blocks(
        source,
        materialized_tokens=4,
    )
    shared_block_id = source.block_table[0]
    before = manager.ownership_generation
    consumer = make_sequence([1, 2, 3, 4, 9])

    manager.allocate(
        consumer,
        publish_hashes=False,
        max_cached_tokens=4,
    )

    assert consumer.block_table[0] == shared_block_id
    assert manager.blocks[shared_block_id].ref_count == 2
    assert manager.ownership_generation > before


def test_reserved_block_release_advances_ownership_generation():
    manager, sequence = make_allocated_fixture()
    before_reserve = manager.ownership_generation

    reserved = manager.reserve_append_blocks(sequence, 8)
    after_reserve = manager.ownership_generation
    manager.release_reserved_blocks(reserved)

    assert reserved
    assert after_reserve > before_reserve
    assert manager.ownership_generation > after_reserve


def test_stable_capture_reuses_identical_seal_without_rescanning(
    monkeypatch,
):
    manager, sequence = make_allocated_fixture()
    calls = 0
    original = manager.block_identities

    def counted(block_ids):
        nonlocal calls
        calls += 1
        return original(block_ids)

    monkeypatch.setattr(manager, "block_identities", counted)
    write_block_index = len(sequence.block_table) - 1

    first = manager.capture_block_table_identity(
        sequence,
        write_block_index=write_block_index,
    )
    second = manager.capture_block_table_identity(
        sequence,
        write_block_index=write_block_index,
    )
    manager.validate_block_table_identity(sequence, second)

    assert second is first
    assert calls == 1


def test_cached_immutable_seal_skips_redundant_digest_validation(
    monkeypatch,
):
    manager, sequence = make_allocated_fixture()
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )

    def forbidden(_identity_sha256):
        raise AssertionError(
            "cached immutable seal digest was already validated"
        )

    monkeypatch.setattr(
        manager,
        "_validate_identity_digest",
        forbidden,
    )

    manager.validate_block_table_identity(sequence, seal)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda sequence: sequence.block_table.append(
            sequence.block_table[-1]
        ),
        lambda sequence: setattr(
            sequence,
            "block_table",
            list(sequence.block_table),
        ),
    ),
    ids=("append", "same_length_replacement"),
)
def test_block_table_mutation_invalidates_seal_before_identity_scan(
    monkeypatch,
    mutate,
):
    manager, sequence = make_allocated_fixture()
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )

    def forbidden(_block_ids):
        raise AssertionError("stale seal must reject before row scan")

    monkeypatch.setattr(manager, "block_identities", forbidden)
    mutate(sequence)

    with pytest.raises(RuntimeError, match="block table.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_table_revision_drift_rejects_seal_before_identity_scan(
    monkeypatch,
):
    manager, sequence = make_allocated_fixture()
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )

    def forbidden(_block_ids):
        raise AssertionError("stale seal must reject before row scan")

    monkeypatch.setattr(manager, "block_identities", forbidden)
    sequence.block_table[-1] = sequence.block_table[-1]

    with pytest.raises(RuntimeError, match="block table.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_predecessor_generation_drift_reports_precise_staleness():
    manager = BlockManager(num_blocks=16, block_size=4)
    sequence = make_sequence(range(1, 10))
    manager.allocate(sequence, publish_hashes=False)
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    predecessor = manager.blocks[sequence.block_table[-2]]

    predecessor.generation += 1

    with pytest.raises(RuntimeError, match="predecessor block.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_unrelated_ownership_change_invalidates_seal():
    manager, sequence = make_allocated_fixture()
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    unrelated = make_sequence([9])

    manager.allocate(unrelated, publish_hashes=False)

    with pytest.raises(RuntimeError, match="ownership.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_recapture_after_ownership_change_scans_once_and_reseals(
    monkeypatch,
):
    manager, sequence = make_allocated_fixture()
    first = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    calls = 0
    original = manager.block_identities

    def counted(block_ids):
        nonlocal calls
        calls += 1
        return original(block_ids)

    monkeypatch.setattr(manager, "block_identities", counted)
    unrelated = make_sequence([9])
    manager.allocate(unrelated, publish_hashes=False)

    second = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    third = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )

    assert second is not first
    assert third is second
    assert calls == 1


def test_direct_write_block_generation_drift_invalidates_seal():
    manager, sequence = make_allocated_fixture()
    write_block_index = len(sequence.block_table) - 1
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=write_block_index,
    )
    write_block = manager.blocks[
        sequence.block_table[write_block_index]
    ]
    before = manager.ownership_generation

    write_block.generation += 1

    assert manager.ownership_generation > before
    with pytest.raises(RuntimeError, match="write block.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_write_block_identity_drift_reports_precise_staleness():
    manager, sequence = make_allocated_fixture()
    unrelated = make_sequence([9])
    manager.allocate(unrelated, publish_hashes=False)
    write_block_index = len(sequence.block_table) - 1
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=write_block_index,
    )

    list.__setitem__(
        sequence.block_table,
        write_block_index,
        unrelated.block_table[-1],
    )

    with pytest.raises(RuntimeError, match="write block.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_direct_interior_block_generation_drift_invalidates_seal():
    manager = BlockManager(num_blocks=16, block_size=4)
    sequence = make_sequence(range(1, 10))
    manager.allocate(sequence, publish_hashes=False)
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    interior = manager.blocks[sequence.block_table[0]]
    before = manager.ownership_generation

    interior.generation += 1

    assert manager.ownership_generation > before
    with pytest.raises(RuntimeError, match="ownership.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_predecessor_block_identity_drift_reports_precise_staleness():
    manager = BlockManager(num_blocks=16, block_size=4)
    sequence = make_sequence(range(1, 10))
    unrelated = make_sequence([11])
    manager.allocate(sequence, publish_hashes=False)
    manager.allocate(unrelated, publish_hashes=False)
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )

    list.__setitem__(
        sequence.block_table,
        len(sequence.block_table) - 2,
        unrelated.block_table[-1],
    )

    with pytest.raises(RuntimeError, match="predecessor block.*stale"):
        manager.validate_block_table_identity(sequence, seal)


def test_block_generation_setter_rejects_invalid_values():
    manager, sequence = make_allocated_fixture()
    block = manager.blocks[sequence.block_table[-1]]

    for value in (True, -1, 1.5):
        with pytest.raises(
            ValueError,
            match="block generation must be a non-negative integer",
        ):
            block.generation = value


def test_restoring_block_generation_still_advances_ownership_epoch():
    manager, sequence = make_allocated_fixture()
    block = manager.blocks[sequence.block_table[-1]]
    generation = block.generation
    before = manager.ownership_generation

    block.generation = generation

    assert block.generation == generation
    assert manager.ownership_generation > before


def test_ownership_generation_exhaustion_fails_before_mutation():
    manager, sequence = make_allocated_fixture()
    manager.ownership_generation = (1 << 63) - 1
    before_table = tuple(sequence.block_table)
    before_used = set(manager.used_block_ids)

    with pytest.raises(
        OverflowError,
        match="block ownership generation exhausted",
    ):
        manager.deallocate(sequence)

    assert tuple(sequence.block_table) == before_table
    assert manager.used_block_ids == before_used


def test_wrong_sequence_and_malformed_seal_are_rejected():
    manager, sequence = make_allocated_fixture()
    seal = manager.capture_block_table_identity(
        sequence,
        write_block_index=len(sequence.block_table) - 1,
    )
    wrong = make_sequence([7])
    manager.allocate(wrong, publish_hashes=False)

    with pytest.raises(ValueError, match="sequence ID"):
        manager.validate_block_table_identity(wrong, seal)

    malformed = replace(seal, identity_sha256="not-a-digest")
    with pytest.raises(ValueError, match="identity digest"):
        manager.validate_block_table_identity(sequence, malformed)
