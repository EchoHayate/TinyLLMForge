from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"


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

config_module = types.ModuleType("tinyvllm.config")
config_module.Config = object
sys.modules["tinyvllm.config"] = config_module


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
scheduler_module = _load_module(
    "tinyvllm.engine.scheduler",
    "tinyvllm/engine/scheduler.py",
)
_load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache",
    "tinyvllm/engine/qwen35_hybrid_prefix_cache.py",
)
ticket_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py",
)
_load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_candidate",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py",
)
source_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_source_publication",
    "tinyvllm/engine/qwen35_hybrid_prefix_source_publication.py",
)
coordinator_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication",
    "tinyvllm/engine/qwen35_hybrid_prefix_engine_publication.py",
)

SamplingParams = sampling_module.SamplingParams
Sequence = sequence_module.Sequence
SequenceStatus = sequence_module.SequenceStatus
HybridStateLease = hybrid_module.HybridStateLease
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
Scheduler = scheduler_module.Scheduler
Qwen35HybridPrefixPublicationPayload = (
    ticket_module.Qwen35HybridPrefixPublicationPayload
)
Qwen35HybridPrefixSourcePublisher = (
    source_module.Qwen35HybridPrefixSourcePublisher
)
Qwen35HybridPrefixEnginePublicationCoordinator = (
    coordinator_module.Qwen35HybridPrefixEnginePublicationCoordinator
)
Sequence.block_size = 2


def _load_engine_method(name):
    source = ENGINE_PATH.read_text()
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method_node.decorator_list = []
    namespace = {
        "Qwen35HybridPrefixPublicationPayload": (
            Qwen35HybridPrefixPublicationPayload
        ),
        "Qwen35HybridPrefixSourcePublisher": (
            Qwen35HybridPrefixSourcePublisher
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method_node], type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


_install_source_publisher = _load_engine_method(
    "install_qwen35_hybrid_prefix_source_publisher"
)
_publish_prefix = _load_engine_method(
    "publish_qwen35_hybrid_prefix"
)
_validate_payloads = _load_engine_method(
    "_validate_hybrid_prefix_publication_payloads"
)


def _config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        max_num_prefill_tokens_per_step=0,
        chunked_prefill_decode_first=True,
        chunked_prefill_max_consecutive_chunks=0,
        chunked_prefill_mixed_batch=False,
        chunked_prefill_mixed_min_prompt_tokens=0,
        chunked_prefill_adaptive_mixed=False,
        chunked_prefill_adaptive_enter_waiting=8,
        chunked_prefill_adaptive_exit_waiting=2,
        chunked_prefill_adaptive_transition_steps=2,
        chunked_prefill_adaptive_max_mixed_steps=2,
        chunked_prefill_slo_mixed=False,
        chunked_prefill_slo_target_gap_ns=0,
        chunked_prefill_slo_reserve_ns=0,
        chunked_prefill_slo_cost_intercept_ns=0,
        chunked_prefill_slo_cost_per_prefill_token_ns=0,
        chunked_prefill_slo_min_chunk_tokens=1,
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=2,
    )


def _sequence():
    return Sequence(
        [1, 2, 3, 4],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
        ),
    )


def _lease(sequence):
    return HybridStateLease(
        sequence.hybrid_state_slot_id,
        sequence.hybrid_state_generation,
        sequence.seq_id,
    )


class _Engine:

    def __init__(self, scheduler, sequence):
        self.scheduler = scheduler
        self.sequence = sequence
        self.model_runner = SimpleNamespace(world_size=1)
        self.qwen35_hybrid_prefix_source_publisher = None
        self.qwen35_hybrid_prefix_source_publisher_hook = None
        self.qwen35_hybrid_prefix_source_publisher_configuration = None
        self.qwen35_hybrid_prefix_engine_publication_coordinator = (
            Qwen35HybridPrefixEnginePublicationCoordinator(
                self,
                timeout_s=0.25,
            )
        )
        self.phase_calls = []
        self.failure_operation = None

    def install_qwen35_hybrid_prefix_source_publisher(self, **kwargs):
        return _install_source_publisher(self, **kwargs)

    def publish_qwen35_hybrid_prefix(self, payloads):
        return _publish_prefix(self, payloads)

    def _validate_hybrid_prefix_publication_payloads(self, payloads):
        return _validate_payloads(self, payloads)

    def _assert_source_live(self, payloads, operation):
        sequence = self.sequence
        assert sequence.completion_token_ids == []
        assert sequence.block_table
        assert sequence.status != SequenceStatus.FINISHED
        self.scheduler.hybrid_state_allocator.validate(_lease(sequence))
        payload = payloads[0]
        assert payload.request_id == sequence.seq_id
        assert payload.token_ids == tuple(sequence.prompt_token_ids)
        assert payload.lease == _lease(sequence)
        assert tuple(
            identity[0] for identity in payload.block_identities
        ) == tuple(sequence.block_table[:2])
        for block_id, generation, block_hash in payload.block_identities:
            block = self.scheduler.block_manager.blocks[block_id]
            assert block.ref_count > 0
            assert block.generation == generation
            assert block.hash == block_hash
        self.phase_calls.append(operation)

    def _phase(self, payloads, operation, success_status):
        self._assert_source_live(payloads, operation)
        status = (
            "error"
            if self.failure_operation == operation
            else success_status
        )
        return ({
            "ticket_id": payloads[0].ticket_id,
            "participant_id": 0,
            "operation": operation,
            "status": status,
            "detail": (
                "injected publication failure"
                if status == "error"
                else ""
            ),
        },)

    def prepare_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        assert timeout_s == 0.25
        return self._phase(payloads, "prepare", "prepared")

    def precommit_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        assert timeout_s == 0.25
        return self._phase(
            payloads,
            "precommit",
            "precommitted",
        )

    def finalize_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        assert timeout_s == 0.25
        return self._phase(payloads, "finalize", "finalized")

    def seal_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        assert timeout_s == 0.25
        return self._phase(payloads, "seal", "committed")

    def rollback_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        assert timeout_s == 0.25
        return self._phase(payloads, "rollback", "rolled_back")


def _fixture():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence()
    engine = _Engine(scheduler, sequence)
    return engine, scheduler, sequence, allocator


def _run_prefill(scheduler, sequence):
    scheduler.add(sequence)
    scheduled = scheduler.schedule()
    scheduler.postprocess(
        scheduled[0],
        [99],
        scheduled[1],
        scheduled[2],
    )


def _expect_error(function, message):
    try:
        function()
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_default_off_prefill_runs_without_publication():
    engine, scheduler, sequence, _ = _fixture()

    _run_prefill(scheduler, sequence)

    assert engine.phase_calls == []
    assert sequence.completion_token_ids == [99]
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []


def test_explicit_install_publishes_once_before_append_and_release():
    engine, scheduler, sequence, _ = _fixture()
    publisher = (
        engine.install_qwen35_hybrid_prefix_source_publisher(
            model_fingerprint="model-a",
            layout_fingerprint="layout-a",
            dtype=torch.float32,
        )
    )

    _run_prefill(scheduler, sequence)

    assert publisher is engine.qwen35_hybrid_prefix_source_publisher
    assert engine.phase_calls == [
        "prepare",
        "precommit",
        "finalize",
        "seal",
    ]
    assert sequence.completion_token_ids == [99]
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert scheduler.hybrid_state_allocator.observation_snapshot()[
        "used_slots"
    ] == 0


def test_publication_failure_stops_before_append_and_release():
    engine, scheduler, sequence, allocator = _fixture()
    engine.failure_operation = "precommit"
    engine.install_qwen35_hybrid_prefix_source_publisher(
        model_fingerprint="model-a",
        layout_fingerprint="layout-a",
        dtype=torch.float32,
    )
    scheduler.add(sequence)
    scheduled = scheduler.schedule()

    _expect_error(
        lambda: scheduler.postprocess(
            scheduled[0],
            [99],
            scheduled[1],
            scheduled[2],
        ),
        "publication precommit failed",
    )

    assert engine.phase_calls == [
        "prepare",
        "precommit",
        "rollback",
    ]
    assert sequence.completion_token_ids == []
    assert sequence.status != SequenceStatus.FINISHED
    assert sequence.block_table
    allocator.validate(_lease(sequence))
    _expect_error(scheduler.schedule, "poisoned")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "explicit prefill publication integration tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
