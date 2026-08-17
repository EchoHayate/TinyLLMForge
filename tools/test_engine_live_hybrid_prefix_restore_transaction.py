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
    "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_restore_ticket.py",
)
live_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore",
    "tinyvllm/engine/qwen35_hybrid_prefix_engine_restore.py",
)

SamplingParams = sampling_module.SamplingParams
Sequence = sequence_module.Sequence
BlockManager = block_module.BlockManager
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
Qwen35HybridPrefixKey = cache_module.Qwen35HybridPrefixKey
Qwen35HybridPrefixEngineRestoreCoordinator = (
    live_module.Qwen35HybridPrefixEngineRestoreCoordinator
)
Sequence.block_size = 4


def _sequence(tokens):
    return Sequence(
        list(tokens),
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
        ),
    )


def _key(block_manager, tokens, world_size=2):
    terminal_hash = -1
    for start in range(0, len(tokens), block_manager.block_size):
        terminal_hash = block_manager.compute_hash(
            list(tokens[start:start + block_manager.block_size]),
            terminal_hash,
        )
    return Qwen35HybridPrefixKey(
        token_hash=terminal_hash,
        token_count=len(tokens),
        terminal_block_hash=terminal_hash,
        block_size=block_manager.block_size,
        model_fingerprint="engine-live-restore-test-model",
        layout_fingerprint="engine-live-restore-test-layout",
        tensor_parallel_size=world_size,
        dtype=torch.float32,
    )


class _Engine:
    def __init__(self, block_manager, allocator, world_size=2):
        self.model_runner = types.SimpleNamespace(world_size=world_size)
        self.scheduler = types.SimpleNamespace(
            block_manager=block_manager,
            hybrid_state_allocator=allocator,
        )
        self.calls = []
        self.prepare_rows = tuple(
            {
                "ticket_id": -1,
                "participant_id": rank,
                "operation": "prepare",
                "status": "prepared",
                "detail": "",
            }
            for rank in range(world_size)
        )
        self.failures = {}
        self.poison_reasons = []

    def _rows(self, payload, operation, status="ok"):
        return tuple(
            {
                "ticket_id": payload.ticket_id,
                "participant_id": rank,
                "operation": operation,
                "status": status,
                "detail": "",
            }
            for rank in range(self.model_runner.world_size)
        )

    def prepare_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.calls.append(("prepare", payload, timeout_s))
        error = self.failures.get("prepare")
        if error is not None:
            raise error
        return tuple(
            {
                **row,
                "ticket_id": payload.ticket_id,
            }
            for row in self.prepare_rows
        )

    def validate_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.calls.append(("validate", payload, timeout_s))
        error = self.failures.get("validate")
        if error is not None:
            raise error
        return self._rows(payload, "validate")

    def commit_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.calls.append(("commit", payload, timeout_s))
        error = self.failures.get("commit")
        if error is not None:
            raise error
        return self._rows(payload, "commit")

    def rollback_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.calls.append(("rollback", payload, timeout_s))
        error = self.failures.get("rollback")
        if error is not None:
            raise error
        return self._rows(payload, "rollback")

    def _poison_model_runner_ack_collector(self, reason):
        self.poison_reasons.append(reason)


def _fixture():
    block_manager = BlockManager(num_blocks=12, block_size=4)
    prefix_tokens = (1, 2, 3, 4)
    source = _sequence(prefix_tokens)
    block_manager.allocate(
        source,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(source, 0, len(source))
    block_manager.deallocate(source)
    allocator = HybridStateSlotAllocator(capacity=4)
    engine = _Engine(block_manager, allocator)
    coordinator = Qwen35HybridPrefixEngineRestoreCoordinator(
        engine,
        block_manager,
        allocator,
        timeout_s=0.25,
    )
    return {
        "block_manager": block_manager,
        "allocator": allocator,
        "engine": engine,
        "coordinator": coordinator,
        "tokens": prefix_tokens,
        "key": _key(block_manager, prefix_tokens),
    }


def _snapshot(fixture):
    allocator_snapshot = fixture[
        "allocator"
    ].observation_snapshot()
    return {
        "free_count": len(
            fixture["block_manager"].free_block_ids
        ),
        "used": tuple(sorted(fixture["block_manager"].used_block_ids)),
        "refs": tuple(
            block.ref_count for block in fixture["block_manager"].blocks
        ),
        "allocator_free_slots": allocator_snapshot["free_slots"],
        "allocator_used_slots": allocator_snapshot["used_slots"],
        "allocator_owners": allocator_snapshot["owners"],
    }


def _assert_pristine(sequence):
    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert sequence.num_computed_tokens == 0
    assert sequence.hybrid_state_slot_id == -1
    assert sequence.hybrid_state_generation == 0


def test_live_restore_keeps_resources_private_until_validate_then_commits():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    engine = fixture["engine"]
    original_prepare = (
        engine.prepare_model_runner_hybrid_prefix_restore
    )

    def inspect_prepare(payload, *, timeout_s):
        ticket = fixture["coordinator"].last_ticket
        assert ticket.state == "reserved"
        assert ticket.reservation.state == "reserved"
        _assert_pristine(destination)
        assert fixture["allocator"].validate(payload.lease) == payload.lease
        return original_prepare(payload, timeout_s=timeout_s)

    engine.prepare_model_runner_hybrid_prefix_restore = inspect_prepare

    restored = fixture["coordinator"].acquire(
        destination,
        fixture["key"],
        fixture["tokens"],
    )

    assert restored is True
    ticket = fixture["coordinator"].last_ticket
    assert ticket.state == "committed"
    assert ticket.reservation.state == "attached"
    assert destination.block_table == list(ticket.reservation.block_ids)
    assert destination.num_cached_tokens == len(fixture["tokens"])
    assert destination.hybrid_state_slot_id == ticket.payload.lease.slot_id
    assert [call[0] for call in engine.calls] == [
        "prepare",
        "validate",
        "commit",
    ]


def test_constructor_rejects_engine_resource_identity_mismatch():
    fixture = _fixture()
    wrong_block_manager = BlockManager(num_blocks=4, block_size=4)
    wrong_allocator = HybridStateSlotAllocator(capacity=4)

    for block_manager, allocator in (
        (wrong_block_manager, fixture["allocator"]),
        (fixture["block_manager"], wrong_allocator),
    ):
        try:
            Qwen35HybridPrefixEngineRestoreCoordinator(
                fixture["engine"],
                block_manager,
                allocator,
                timeout_s=0.25,
            )
        except ValueError as error:
            assert "Scheduler" in str(error)
        else:
            raise AssertionError(
                "Engine resource identity mismatch was accepted"
            )


def test_exact_prefix_miss_releases_private_kv_without_allocating_lease():
    fixture = _fixture()
    destination = _sequence([8, 7, 6, 5, 9])
    key = _key(fixture["block_manager"], (8, 7, 6, 5))
    before = _snapshot(fixture)

    restored = fixture["coordinator"].acquire(
        destination,
        key,
        (8, 7, 6, 5),
    )

    assert restored is False
    assert _snapshot(fixture) == before
    _assert_pristine(destination)
    assert fixture["engine"].calls == []


def test_prepare_miss_broadcasts_rollback_and_releases_private_resources():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    before = _snapshot(fixture)
    fixture["engine"].prepare_rows = (
        {
            "ticket_id": -1,
            "participant_id": 0,
            "operation": "prepare",
            "status": "prepared",
            "detail": "",
        },
        {
            "ticket_id": -1,
            "participant_id": 1,
            "operation": "prepare",
            "status": "miss",
            "detail": "snapshot miss",
        },
    )

    restored = fixture["coordinator"].acquire(
        destination,
        fixture["key"],
        fixture["tokens"],
    )

    assert restored is False
    assert fixture["coordinator"].last_ticket.state == "rolled_back"
    assert [call[0] for call in fixture["engine"].calls] == [
        "prepare",
        "rollback",
    ]
    assert _snapshot(fixture) == before
    _assert_pristine(destination)


def test_validate_failure_rolls_back_and_cleanup_failure_poisons():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    fixture["engine"].failures["validate"] = RuntimeError(
        "validate transport failed"
    )

    try:
        fixture["coordinator"].acquire(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "validate transport failed" in str(error)
    else:
        raise AssertionError("validate failure was swallowed")
    assert fixture["coordinator"].last_ticket.state == "rolled_back"
    _assert_pristine(destination)

    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    fixture["engine"].prepare_rows = (
        {
            "ticket_id": -1,
            "participant_id": 0,
            "operation": "prepare",
            "status": "prepared",
            "detail": "",
        },
        {
            "ticket_id": -1,
            "participant_id": 1,
            "operation": "prepare",
            "status": "error",
            "detail": "rank error",
        },
    )
    fixture["engine"].failures["rollback"] = RuntimeError(
        "rollback transport failed"
    )
    try:
        fixture["coordinator"].acquire(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "rollback" in str(error)
    else:
        raise AssertionError("rollback failure was swallowed")
    assert fixture["coordinator"].last_ticket.state == "rollback_failed"
    failed_ticket = fixture["coordinator"].last_ticket
    assert failed_ticket.reservation.state == "reserved"
    assert fixture["allocator"].validate(
        failed_ticket.payload.lease
    ) == failed_ticket.payload.lease
    try:
        fixture["coordinator"].acquire(
            _sequence([1, 2, 3, 4, 8]),
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "poisoned" in str(error)
    else:
        raise AssertionError("poisoned coordinator accepted reuse")


def test_precommit_stale_local_state_rolls_back_before_publication():
    for scenario in ("allocator", "reservation", "sequence"):
        fixture = _fixture()
        destination = _sequence([1, 2, 3, 4, 9])
        original_prepare = (
            fixture[
                "engine"
            ].prepare_model_runner_hybrid_prefix_restore
        )

        def mutate_after_prepare(payload, *, timeout_s):
            rows = original_prepare(payload, timeout_s=timeout_s)
            ticket = fixture["coordinator"].last_ticket
            if scenario == "allocator":
                fixture["allocator"].release(payload.lease)
            elif scenario == "reservation":
                block_id = ticket.reservation.block_ids[0]
                fixture["block_manager"].blocks[
                    block_id
                ].generation += 1
            else:
                destination.hybrid_state_generation = 99
            return rows

        fixture[
            "engine"
        ].prepare_model_runner_hybrid_prefix_restore = (
            mutate_after_prepare
        )
        try:
            fixture["coordinator"].acquire(
                destination,
                fixture["key"],
                fixture["tokens"],
            )
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"stale {scenario} precommit state was accepted"
            )

        ticket = fixture["coordinator"].last_ticket
        assert ticket.state == "rolled_back"
        assert ticket.reservation.state == "released"
        assert destination.block_table == []
        assert destination.num_cached_tokens == 0
        if scenario != "sequence":
            assert destination.hybrid_state_slot_id == -1
            assert destination.hybrid_state_generation == 0


def test_attach_then_raise_is_post_publication_failure_and_poisons():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    original_attach = (
        fixture["block_manager"].attach_sequence_reservation
    )

    def attach_then_raise(reservation, sequence):
        original_attach(reservation, sequence)
        raise RuntimeError("injected post-attach failure")

    fixture["block_manager"].attach_sequence_reservation = (
        attach_then_raise
    )
    try:
        fixture["coordinator"].acquire(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "post-attach" in str(error)
    else:
        raise AssertionError("post-attach failure was swallowed")

    ticket = fixture["coordinator"].last_ticket
    assert ticket.state == "commit_failed"
    assert ticket.reservation.state == "attached"
    assert destination.block_table == list(ticket.reservation.block_ids)
    assert fixture["engine"].poison_reasons
    assert [call[0] for call in fixture["engine"].calls] == [
        "prepare",
        "validate",
    ]


def test_commit_failure_after_publication_poisons_and_preserves_ownership():
    fixture = _fixture()
    destination = _sequence([1, 2, 3, 4, 9])
    fixture["engine"].failures["commit"] = RuntimeError(
        "commit transport failed"
    )

    try:
        fixture["coordinator"].acquire(
            destination,
            fixture["key"],
            fixture["tokens"],
        )
    except RuntimeError as error:
        assert "commit transport failed" in str(error)
    else:
        raise AssertionError("commit failure was swallowed")

    ticket = fixture["coordinator"].last_ticket
    assert ticket.state == "commit_failed"
    assert ticket.reservation.state == "attached"
    assert destination.block_table == list(ticket.reservation.block_ids)
    assert fixture["allocator"].validate(ticket.payload.lease) == (
        ticket.payload.lease
    )
    assert fixture["engine"].poison_reasons


def _load_engine_method(name):
    path = ROOT / "tinyvllm/engine/llm_engine.py"
    tree = ast.parse(path.read_text(), filename=str(path))
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
    module = ast.Module(body=[method_node], type_ignores=[])
    namespace = {
        "Qwen35HybridPrefixEngineRestoreCoordinator": (
            Qwen35HybridPrefixEngineRestoreCoordinator
        ),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def test_engine_installation_and_delegation_are_explicit_and_fail_closed():
    fixture = _fixture()
    install = _load_engine_method(
        "install_qwen35_hybrid_prefix_engine_restore_coordinator"
    )
    acquire = _load_engine_method("acquire_qwen35_hybrid_prefix")
    engine = types.SimpleNamespace(
        scheduler=fixture["engine"].scheduler,
        model_runner=types.SimpleNamespace(world_size=2),
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
    )
    destination = _sequence([1, 2, 3, 4, 9])

    try:
        acquire(engine, destination, fixture["key"], fixture["tokens"])
    except RuntimeError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("uninstalled Engine acquisition was accepted")

    fixture["coordinator"].engine = engine
    install(engine, fixture["coordinator"])
    install(engine, fixture["coordinator"])
    delegated = []
    fixture["coordinator"].acquire = (
        lambda *args: delegated.append(args) or True
    )
    assert acquire(
        engine,
        destination,
        fixture["key"],
        fixture["tokens"],
    ) is True
    assert delegated == [
        (destination, fixture["key"], fixture["tokens"])
    ]

    replacement = Qwen35HybridPrefixEngineRestoreCoordinator(
        engine,
        engine.scheduler.block_manager,
        engine.scheduler.hybrid_state_allocator,
        timeout_s=0.25,
    )
    try:
        install(engine, replacement)
    except RuntimeError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError("coordinator replacement was accepted")

    wrong_engine = types.SimpleNamespace(
        scheduler=engine.scheduler,
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
    )
    try:
        install(wrong_engine, fixture["coordinator"])
    except ValueError as error:
        assert "this LLMEngine" in str(error)
    else:
        raise AssertionError("wrong Engine coordinator was installed")

    for invalid in (object(),):
        fresh = types.SimpleNamespace(
            scheduler=engine.scheduler,
            qwen35_hybrid_prefix_engine_restore_coordinator=None,
        )
        try:
            install(fresh, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid coordinator was installed")


def test_scheduler_guard_and_engine_step_do_not_use_live_coordinator():
    scheduler_source = (
        ROOT / "tinyvllm/engine/scheduler.py"
    ).read_text()
    assert (
        "hybrid prefix reuse requires aligned state snapshot"
        in scheduler_source
    )
    step = ast.get_source_segment(
        (ROOT / "tinyvllm/engine/llm_engine.py").read_text(),
        next(
            node
            for node in next(
                node
                for node in ast.parse(
                    (
                        ROOT / "tinyvllm/engine/llm_engine.py"
                    ).read_text()
                ).body
                if isinstance(node, ast.ClassDef)
                and node.name == "LLMEngine"
            ).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "step"
        ),
    )
    assert "qwen35_hybrid_prefix_engine_restore" not in step


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "engine live hybrid prefix restore transaction tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
