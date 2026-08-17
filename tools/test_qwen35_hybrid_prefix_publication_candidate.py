from __future__ import annotations

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
cache_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache",
    "tinyvllm/engine/qwen35_hybrid_prefix_cache.py",
)
ticket_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py",
)
candidate_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_candidate",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py",
)

SamplingParams = sampling_module.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_module.BlockManager
HybridStateLease = hybrid_module.HybridStateLease
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
Qwen35HybridPrefixPublicationCandidate = (
    candidate_module.Qwen35HybridPrefixPublicationCandidate
)
capture_candidate = (
    candidate_module
    .capture_qwen35_hybrid_prefix_publication_candidate
)
Sequence.block_size = 4


def _sequence(tokens):
    return Sequence(
        list(tokens),
        SamplingParams(
            temperature=0.0,
            max_tokens=4,
            ignore_eos=True,
        ),
    )


def _fixture(tokens=(1, 2, 3, 4, 5, 6, 7, 8)):
    block_manager = BlockManager(num_blocks=8, block_size=4)
    allocator = HybridStateSlotAllocator(capacity=2)
    sequence = _sequence(tokens)
    lease = allocator.allocate(sequence.seq_id)
    sequence.hybrid_state_slot_id = lease.slot_id
    sequence.hybrid_state_generation = lease.generation
    block_manager.allocate(
        sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(
        sequence,
        0,
        sequence.num_prompt_tokens,
    )
    sequence.num_computed_tokens = sequence.num_prompt_tokens
    return block_manager, allocator, sequence, lease


def _capture(block_manager, allocator, sequence):
    return capture_candidate(
        sequence,
        block_manager,
        allocator,
        model_fingerprint="model-a",
        layout_fingerprint="layout-a",
        tensor_parallel_size=2,
        dtype=torch.float32,
    )


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_capture_exact_aligned_completed_prompt():
    block_manager, allocator, sequence, lease = _fixture()
    sequence.append_token(99)

    candidate = _capture(block_manager, allocator, sequence)

    assert isinstance(
        candidate,
        Qwen35HybridPrefixPublicationCandidate,
    )
    assert candidate.request_id == sequence.seq_id
    assert candidate.token_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert candidate.lease == lease
    assert candidate.key.token_count == 8
    assert candidate.key.block_size == 4
    assert candidate.key.tensor_parallel_size == 2
    assert candidate.key.dtype == torch.float32
    assert candidate.key.model_fingerprint == "model-a"
    assert candidate.key.layout_fingerprint == "layout-a"
    assert candidate.key.token_hash == candidate.key.terminal_block_hash
    assert candidate.block_identities == tuple(
        (
            block_id,
            block_manager.blocks[block_id].generation,
            block_manager.blocks[block_id].hash,
        )
        for block_id in sequence.block_table[:2]
    )


def test_capture_is_immutable_against_source_mutation():
    block_manager, allocator, sequence, _ = _fixture()
    candidate = _capture(block_manager, allocator, sequence)
    token_ids = candidate.token_ids
    block_identities = candidate.block_identities

    sequence.token_ids[0] = 77
    sequence.block_table[0] = sequence.block_table[-1]
    block_manager.blocks[block_identities[0][0]].generation += 1

    assert candidate.token_ids == token_ids
    assert candidate.block_identities == block_identities


def test_capture_rejects_ineligible_or_stale_sources():
    block_manager, allocator, sequence, lease = _fixture(
        tokens=(1, 2, 3, 4, 5),
    )
    candidate = _capture(block_manager, allocator, sequence)
    assert candidate.key.token_count == 4
    assert candidate.token_ids == (1, 2, 3, 4)
    assert len(candidate.block_identities) == 1

    block_manager, allocator, sequence, lease = _fixture()
    sequence.num_computed_tokens = 4
    _expect_error(
        lambda: _capture(block_manager, allocator, sequence),
        "computed",
    )

    block_manager, allocator, sequence, lease = _fixture()
    block_manager.blocks[sequence.block_table[1]].hash = -1
    _expect_error(
        lambda: _capture(block_manager, allocator, sequence),
        "hash",
    )

    block_manager, allocator, sequence, lease = _fixture()
    stale_block_id = sequence.block_table[0]
    block_manager.used_block_ids.remove(stale_block_id)
    block_manager.free_block_ids.append(stale_block_id)
    block_manager.blocks[stale_block_id].ref_count = 0
    _expect_error(
        lambda: _capture(block_manager, allocator, sequence),
        "generation",
    )

    block_manager, allocator, sequence, lease = _fixture()
    block_manager.blocks[sequence.block_table[0]].token_ids[0] = 77
    _expect_error(
        lambda: _capture(block_manager, allocator, sequence),
        "token",
    )

    block_manager, allocator, sequence, lease = _fixture()
    allocator.release(lease)
    _expect_error(
        lambda: _capture(block_manager, allocator, sequence),
        "lease",
    )


def test_candidate_builds_exact_payload_matrix():
    block_manager, allocator, sequence, lease = _fixture()
    candidate = _capture(block_manager, allocator, sequence)

    payloads = candidate.publication_payloads(
        ticket_id=11,
        world_size=2,
    )

    assert tuple(
        payload.participant_id for payload in payloads
    ) == (0, 1)
    for payload in payloads:
        assert payload.ticket_id == 11
        assert payload.request_id == sequence.seq_id
        assert payload.key == candidate.key
        assert payload.token_ids == candidate.token_ids
        assert (
            payload.block_identities
            == candidate.block_identities
        )
        assert payload.lease == lease
    assert pickle.loads(pickle.dumps(payloads)) == payloads

    _expect_error(
        lambda: candidate.publication_payloads(
            ticket_id=-1,
            world_size=2,
        ),
        "ticket_id",
    )
    _expect_error(
        lambda: candidate.publication_payloads(
            ticket_id=12,
            world_size=1,
        ),
        "world_size",
    )


def test_candidate_revalidates_exact_live_source():
    block_manager, allocator, sequence, _ = _fixture()
    candidate = _capture(block_manager, allocator, sequence)

    assert candidate.validate_source(
        sequence,
        block_manager,
        allocator,
    ) is candidate

    sequence.token_ids[0] = 77
    _expect_error(
        lambda: candidate.validate_source(
            sequence,
            block_manager,
            allocator,
        ),
        "changed",
    )


def test_candidate_module_is_not_runtime_wired():
    forbidden = (
        "capture_qwen35_hybrid_prefix_publication_candidate"
    )
    for relative_path in (
        "tinyvllm/engine/llm_engine.py",
        "tinyvllm/engine/scheduler.py",
        "tinyvllm/engine/model_runner.py",
    ):
        source = (ROOT / relative_path).read_text()
        assert forbidden not in source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 hybrid prefix publication candidate tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
