from __future__ import annotations

import ast
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
publication_ticket_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py",
)
engine_restore_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore",
    "tinyvllm/engine/qwen35_hybrid_prefix_engine_restore.py",
)
owner_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_owner",
    "tinyvllm/engine/qwen35_hybrid_prefix_owner.py",
)

BlockManager = block_module.BlockManager
HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateRuntimeBridge = hybrid_module.HybridStateRuntimeBridge
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35HybridPrefixRestoreParticipant = (
    ticket_module.Qwen35HybridPrefixRestoreParticipant
)
Qwen35HybridPrefixPublicationParticipant = (
    publication_ticket_module.Qwen35HybridPrefixPublicationParticipant
)
Qwen35HybridPrefixEngineRestoreCoordinator = (
    engine_restore_module.Qwen35HybridPrefixEngineRestoreCoordinator
)
Qwen35HybridPrefixRestoreOwner = (
    owner_module.Qwen35HybridPrefixRestoreOwner
)
build_qwen35_hybrid_prefix_restore_owner = (
    owner_module.build_qwen35_hybrid_prefix_restore_owner
)


def _layout():
    return HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (2, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            2,
            "linear_convolution",
            (2, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            2,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))


def _pool(capacity=4):
    return HybridStateTensorPool(
        _layout(),
        capacity=capacity,
        device="cpu",
    )


def test_owner_factory_reuses_exact_pool_storage_and_builds_coherent_graph():
    pool = _pool()
    storage = {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    }

    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=1,
        max_entries=8,
        max_bytes=1 << 20,
    )

    assert isinstance(owner, Qwen35HybridPrefixRestoreOwner)
    assert owner.pool is pool
    assert tuple(
        adapter.layer_index for adapter in owner.adapters
    ) == (0, 2)
    assert all(adapter.pool is pool for adapter in owner.adapters)
    assert owner.state_transaction.adapters == owner.adapters
    assert owner.state_transaction.pool is pool
    assert owner.snapshot_cache.state_transaction is (
        owner.state_transaction
    )
    assert owner.participant.pool is pool
    assert owner.participant.snapshot_cache is owner.snapshot_cache
    assert owner.participant.participant_id == 1
    assert isinstance(
        owner.publication_participant,
        Qwen35HybridPrefixPublicationParticipant,
    )
    assert owner.publication_participant.pool is pool
    assert owner.publication_participant.snapshot_cache is (
        owner.snapshot_cache
    )
    assert owner.publication_participant.participant_id == 1
    assert owner.max_entries == 8
    assert owner.max_bytes == 1 << 20
    assert {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    } == storage


def test_owner_factory_rejects_invalid_or_incomplete_pool_layouts():
    invalid_layouts = (
        HybridStateLayout((
            HybridStateComponentSpec(
                0,
                "linear_convolution",
                (1,),
                torch.float32,
            ),
        )),
        HybridStateLayout((
            HybridStateComponentSpec(
                0,
                "linear_recurrent",
                (1,),
                torch.float32,
            ),
        )),
    )
    for layout in invalid_layouts:
        pool = HybridStateTensorPool(layout, 1, "cpu")
        try:
            build_qwen35_hybrid_prefix_restore_owner(
                pool,
                participant_id=0,
                max_entries=1,
                max_bytes=1024,
            )
        except ValueError as error:
            assert "complete" in str(error)
        else:
            raise AssertionError("incomplete owner layout was accepted")

    for kwargs in (
        {"pool": object(), "participant_id": 0,
         "max_entries": 1, "max_bytes": 1024},
        {"pool": _pool(), "participant_id": -1,
         "max_entries": 1, "max_bytes": 1024},
        {"pool": _pool(), "participant_id": 0,
         "max_entries": 0, "max_bytes": 1024},
        {"pool": _pool(), "participant_id": 0,
         "max_entries": 1, "max_bytes": 0},
    ):
        try:
            build_qwen35_hybrid_prefix_restore_owner(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid owner factory input was accepted")


def _load_class_method(
    relative_path,
    class_name,
    method_name,
    namespace,
):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace[method_name]


def _runner_method(name):
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
        {
            "Qwen35HybridPrefixRestoreParticipant": (
                Qwen35HybridPrefixRestoreParticipant
            ),
            "Qwen35HybridPrefixPublicationParticipant": (
                Qwen35HybridPrefixPublicationParticipant
            ),
            "Qwen35HybridPrefixRestoreOwner": (
                Qwen35HybridPrefixRestoreOwner
            ),
            "build_qwen35_hybrid_prefix_restore_owner": (
                build_qwen35_hybrid_prefix_restore_owner
            ),
        },
    )


def _runner(pool=None, rank=1):
    runner = types.SimpleNamespace(
        rank=rank,
        hybrid_state_runtime_bridge=(
            HybridStateRuntimeBridge(pool)
            if pool is not None
            else None
        ),
        qwen35_hybrid_prefix_restore_participant=None,
        qwen35_hybrid_prefix_publication_participant=None,
        qwen35_hybrid_prefix_restore_owner=None,
    )
    install_restore = _runner_method(
        "install_qwen35_hybrid_prefix_restore_participant"
    )
    install_publication = _runner_method(
        "install_qwen35_hybrid_prefix_publication_participant"
    )
    runner.install_qwen35_hybrid_prefix_restore_participant = (
        lambda participant: install_restore(runner, participant)
    )
    runner.install_qwen35_hybrid_prefix_publication_participant = (
        lambda participant: install_publication(runner, participant)
    )
    return runner


def test_model_runner_configures_owner_from_bridge_and_is_idempotent():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    pool = _pool()
    runner = _runner(pool)

    first = configure(runner, 8, 1 << 20)
    owner = runner.qwen35_hybrid_prefix_restore_owner
    second = configure(runner, 8, 1 << 20)

    assert owner.pool is pool
    assert runner.qwen35_hybrid_prefix_restore_participant is (
        owner.participant
    )
    assert runner.qwen35_hybrid_prefix_publication_participant is (
        owner.publication_participant
    )
    assert (
        owner.participant.snapshot_cache
        is owner.publication_participant.snapshot_cache
        is owner.snapshot_cache
    )
    assert first == second == {
        "participant_id": 1,
        "capacity": 4,
        "layout_fingerprint": pool.layout.fingerprint,
        "bytes_per_slot": pool.layout.bytes_per_slot,
        "max_entries": 8,
        "max_bytes": 1 << 20,
        "representation": "exact_restore",
        "representation_version": "qwen35_hybrid_prefix_exact_v1",
        "codec": None,
    }
    assert pickle.loads(pickle.dumps(first)) == first
    assert runner.qwen35_hybrid_prefix_restore_owner is owner


def test_model_runner_configuration_fails_closed_without_bridge_or_on_change():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    runner = _runner()
    try:
        configure(runner, 8, 1 << 20)
    except RuntimeError as error:
        assert "runtime bridge" in str(error)
    else:
        raise AssertionError("owner configured without runtime bridge")

    runner = _runner(_pool())
    configure(runner, 8, 1 << 20)
    for limits in ((9, 1 << 20), (8, (1 << 20) + 1)):
        try:
            configure(runner, *limits)
        except RuntimeError as error:
            assert "already configured" in str(error)
        else:
            raise AssertionError("owner configuration changed in place")


class _WorkerAck:
    def __init__(self, rank, result):
        self.rank = rank
        self.result = result


class _Collector:
    def __init__(self):
        self.poison_reasons = []

    def poison(self, reason):
        self.poison_reasons.append(reason)


def _owner_row(rank, *, capacity=4, fingerprint="layout-a",
               bytes_per_slot=1024, max_entries=8, max_bytes=1 << 20,
               representation="exact_restore",
               representation_version="qwen35_hybrid_prefix_exact_v1",
               codec=None):
    return {
        "participant_id": rank,
        "capacity": capacity,
        "layout_fingerprint": fingerprint,
        "bytes_per_slot": bytes_per_slot,
        "max_entries": max_entries,
        "max_bytes": max_bytes,
        "representation": representation,
        "representation_version": representation_version,
        "codec": codec,
    }


def _engine(world_size=2, capacity=4):
    block_manager = BlockManager(12, 4)
    allocator = HybridStateSlotAllocator(capacity)
    scheduler = types.SimpleNamespace(
        block_manager=block_manager,
        hybrid_state_allocator=allocator,
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=world_size),
        scheduler=scheduler,
        model_runner_ack_collector=_Collector(),
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
        qwen35_hybrid_prefix_restore_configuration=None,
    )
    rows = tuple(_owner_row(rank) for rank in range(world_size))
    calls = []

    def acknowledged(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        return (
            rows[0],
            tuple(
                _WorkerAck(rank, rows[rank])
                for rank in range(1, world_size)
            ),
        )

    engine.call_model_runner_acknowledged = acknowledged
    poison = _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_poison_model_runner_ack_collector",
        {},
    )
    install = _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "install_qwen35_hybrid_prefix_engine_restore_coordinator",
        {
            "Qwen35HybridPrefixEngineRestoreCoordinator": (
                Qwen35HybridPrefixEngineRestoreCoordinator
            ),
        },
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    engine.install_qwen35_hybrid_prefix_engine_restore_coordinator = (
        lambda coordinator: install(engine, coordinator)
    )
    engine._rows = rows
    engine._calls = calls
    return engine


def _engine_configure_method():
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "configure_qwen35_hybrid_prefix_restore",
        {
            "Qwen35HybridPrefixEngineRestoreCoordinator": (
                Qwen35HybridPrefixEngineRestoreCoordinator
            ),
        },
    )


def test_engine_configures_all_ranks_then_installs_coordinator():
    configure = _engine_configure_method()
    engine = _engine()

    coordinator = configure(
        engine,
        max_entries=8,
        max_bytes=1 << 20,
        timeout_s=0.5,
    )
    repeated = configure(
        engine,
        max_entries=8,
        max_bytes=1 << 20,
        timeout_s=0.5,
    )

    assert isinstance(
        coordinator,
        Qwen35HybridPrefixEngineRestoreCoordinator,
    )
    assert coordinator is repeated
    assert coordinator.engine is engine
    assert coordinator.block_manager is engine.scheduler.block_manager
    assert coordinator.state_allocator is (
        engine.scheduler.hybrid_state_allocator
    )
    assert coordinator.timeout_s == 0.5
    assert engine._calls == [(
        "configure_qwen35_hybrid_prefix_restore_owner",
        (8, 1 << 20, "exact_restore"),
        0.5,
    )]


def test_engine_owner_identity_mismatch_poisons_and_does_not_install():
    configure = _engine_configure_method()
    scenarios = (
        (
            _owner_row(0),
            _owner_row(0),
        ),
        (
            _owner_row(0),
            _owner_row(1, capacity=5),
        ),
        (
            _owner_row(0),
            _owner_row(1, fingerprint="layout-b"),
        ),
        (
            _owner_row(0),
            _owner_row(1, bytes_per_slot=2048),
        ),
    )
    for rows in scenarios:
        engine = _engine()

        def acknowledged(method_name, *args, timeout_s):
            return rows[0], (_WorkerAck(1, rows[1]),)

        engine.call_model_runner_acknowledged = acknowledged
        try:
            configure(
                engine,
                max_entries=8,
                max_bytes=1 << 20,
                timeout_s=0.5,
            )
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("inconsistent owner identity was accepted")
        assert engine.model_runner_ack_collector.poison_reasons
        assert (
            engine.qwen35_hybrid_prefix_engine_restore_coordinator
            is None
        )


def test_engine_rejects_missing_allocator_capacity_mismatch_and_reconfigure():
    configure = _engine_configure_method()
    engine = _engine(capacity=3)
    try:
        configure(
            engine,
            max_entries=8,
            max_bytes=1 << 20,
            timeout_s=0.5,
        )
    except RuntimeError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("allocator capacity mismatch was accepted")

    engine = _engine()
    engine.scheduler.hybrid_state_allocator = None
    try:
        configure(
            engine,
            max_entries=8,
            max_bytes=1 << 20,
            timeout_s=0.5,
        )
    except RuntimeError as error:
        assert "allocator" in str(error)
    else:
        raise AssertionError("missing allocator was accepted")

    engine = _engine()
    configure(
        engine,
        max_entries=8,
        max_bytes=1 << 20,
        timeout_s=0.5,
    )
    try:
        configure(
            engine,
            max_entries=9,
            max_bytes=1 << 20,
            timeout_s=0.5,
        )
    except RuntimeError as error:
        assert "already configured" in str(error)
    else:
        raise AssertionError("Engine owner configuration changed")


def test_engine_configuration_validates_public_inputs_before_conversion():
    configure = _engine_configure_method()
    invalid = (
        {"max_entries": None, "max_bytes": 1024, "timeout_s": 0.5},
        {"max_entries": 8, "max_bytes": None, "timeout_s": 0.5},
        {"max_entries": 8, "max_bytes": 1024, "timeout_s": None},
        {"max_entries": True, "max_bytes": 1024, "timeout_s": 0.5},
        {"max_entries": 8, "max_bytes": True, "timeout_s": 0.5},
        {"max_entries": 8, "max_bytes": 1024, "timeout_s": True},
    )
    for kwargs in invalid:
        engine = _engine()
        try:
            configure(engine, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid Engine configuration was accepted: {kwargs}"
            )
        assert engine._calls == []


def test_scheduler_guard_and_step_remain_unwired():
    scheduler_source = (
        ROOT / "tinyvllm/engine/scheduler.py"
    ).read_text()
    assert (
        "hybrid prefix reuse requires aligned state snapshot"
        in scheduler_source
    )
    engine_source = (
        ROOT / "tinyvllm/engine/llm_engine.py"
    ).read_text()
    tree = ast.parse(engine_source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    step_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    step_source = ast.get_source_segment(engine_source, step_node)
    assert "configure_qwen35_hybrid_prefix_restore" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 live restore owner factory tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
