from __future__ import annotations

import ast
import hashlib
import importlib.util
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

sampling = _load_module(
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
_load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache",
    "tinyvllm/engine/qwen35_hybrid_prefix_cache.py",
)
_load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py",
)
_load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_candidate",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py",
)
publisher_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_source_publication",
    "tinyvllm/engine/qwen35_hybrid_prefix_source_publication.py",
)

SamplingParams = sampling.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_module.BlockManager
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
Qwen35HybridPrefixSourcePublisher = (
    publisher_module.Qwen35HybridPrefixSourcePublisher
)
Sequence.block_size = 4


class _Engine:
    def __init__(self, block_manager, allocator, world_size=2):
        self.model_runner = types.SimpleNamespace(
            world_size=world_size,
        )
        self.scheduler = types.SimpleNamespace(
            block_manager=block_manager,
            hybrid_state_allocator=allocator,
        )
        self.qwen35_hybrid_prefix_engine_publication_coordinator = object()
        self.payloads = []
        self.result = True
        self.on_publish = None

    def publish_qwen35_hybrid_prefix(self, payloads):
        self.payloads.append(payloads)
        if self.on_publish is not None:
            return self.on_publish(payloads)
        return self.result


def _fixture():
    block_manager = BlockManager(8, 4)
    allocator = HybridStateSlotAllocator(2)
    sequence = Sequence(
        [1, 2, 3, 4, 5, 6, 7, 8],
        SamplingParams(
            temperature=0.0,
            max_tokens=4,
            ignore_eos=True,
        ),
    )
    lease = allocator.allocate(sequence.seq_id)
    sequence.hybrid_state_slot_id = lease.slot_id
    sequence.hybrid_state_generation = lease.generation
    block_manager.allocate(
        sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(sequence, 0, 8)
    sequence.num_computed_tokens = 8
    engine = _Engine(block_manager, allocator)
    publisher = Qwen35HybridPrefixSourcePublisher(
        engine,
        model_fingerprint="model-a",
        layout_fingerprint="layout-a",
        dtype=torch.float32,
    )
    return engine, publisher, sequence


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_source_publisher_builds_monotonic_exact_transactions():
    engine, publisher, sequence = _fixture()

    assert publisher.publish(sequence) is True
    engine.result = False
    assert publisher.publish(sequence) is False

    assert len(engine.payloads) == 2
    assert tuple(
        payload.ticket_id for payload in engine.payloads[0]
    ) == (0, 0)
    assert tuple(
        payload.ticket_id for payload in engine.payloads[1]
    ) == (1, 1)
    assert tuple(
        payload.participant_id for payload in engine.payloads[0]
    ) == (0, 1)


def test_source_publisher_fails_before_dispatch_on_invalid_source():
    engine, publisher, sequence = _fixture()
    sequence.num_computed_tokens = 4

    _expect_error(
        lambda: publisher.publish(sequence),
        "computed",
    )
    assert engine.payloads == []


def test_source_publisher_validates_engine_owners_and_coordinator():
    engine, publisher, sequence = _fixture()
    engine.qwen35_hybrid_prefix_engine_publication_coordinator = None
    _expect_error(
        lambda: publisher.publish(sequence),
        "coordinator",
    )
    assert engine.payloads == []

    engine, publisher, sequence = _fixture()
    engine.scheduler.hybrid_state_allocator = None
    _expect_error(
        lambda: publisher.publish(sequence),
        "allocator",
    )
    assert engine.payloads == []


def test_source_publisher_rejects_reentrant_call():
    engine, publisher, sequence = _fixture()

    def reenter(_):
        return publisher.publish(sequence)

    engine.on_publish = reenter
    _expect_error(
        lambda: publisher.publish(sequence),
        "already active",
    )


def test_source_publisher_rejects_drift_before_dispatch():
    engine, publisher, sequence = _fixture()
    original_capture = (
        publisher_module
        .capture_qwen35_hybrid_prefix_publication_candidate
    )

    def capture_then_mutate(*args, **kwargs):
        candidate = original_capture(*args, **kwargs)
        sequence.token_ids[0] = 77
        return candidate

    publisher_module.capture_qwen35_hybrid_prefix_publication_candidate = (
        capture_then_mutate
    )
    try:
        _expect_error(
            lambda: publisher.publish(sequence),
            "changed",
        )
    finally:
        publisher_module.capture_qwen35_hybrid_prefix_publication_candidate = (
            original_capture
        )
    assert engine.payloads == []


def test_source_publisher_resets_after_transaction_failure():
    engine, publisher, sequence = _fixture()

    def fail(_):
        raise RuntimeError("injected publication failure")

    engine.on_publish = fail
    _expect_error(
        lambda: publisher.publish(sequence),
        "publication failure",
    )
    engine.on_publish = None
    assert publisher.publish(sequence) is True
    assert tuple(
        payload.ticket_id for payload in engine.payloads[-1]
    ) == (1, 1)


def test_source_publisher_is_not_runtime_wired():
    engine_source = (
        ROOT / "tinyvllm/engine/llm_engine.py"
    ).read_text()
    engine_tree = ast.parse(engine_source)
    engine_class = next(
        node for node in engine_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    step = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )
    assert "Qwen35HybridPrefixSourcePublisher" not in ast.unparse(step)
    assert "qwen35_hybrid_prefix_source_publisher" not in ast.unparse(step)
    for relative_path in (
        "tinyvllm/engine/scheduler.py",
        "tinyvllm/engine/model_runner.py",
    ):
        source = (ROOT / relative_path).read_text()
        assert "Qwen35HybridPrefixSourcePublisher" not in source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 source-bound publication tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
