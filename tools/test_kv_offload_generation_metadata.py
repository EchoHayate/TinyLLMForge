from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm" / "engine" / "model_runner.py"
H2D_DIAGNOSTIC_PATH = (
    ROOT / "tinyvllm" / "engine" / "h2d_slot_reuse_diagnostic.py"
)


def _load_h2d_diagnostic_type():
    spec = importlib.util.spec_from_file_location(
        "h2d_slot_reuse_diagnostic_for_generation_metadata_test",
        H2D_DIAGNOSTIC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.H2DSlotReuseDiagnostic


H2DSlotReuseDiagnostic = _load_h2d_diagnostic_type()


def _load_kv_offload_type():
    tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KVOffloadMVP0"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "H2DSlotReuseDiagnostic": H2DSlotReuseDiagnostic,
        "time": time,
    }
    exec(compile(module, str(MODEL_RUNNER_PATH), "exec"), namespace)
    return namespace["KVOffloadMVP0"]


KVOffloadMVP0 = _load_kv_offload_type()


def _load_model_runner_method(method_name, namespace):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    compiled_namespace = dict(namespace)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[function],
                    type_ignores=[],
                )
            ),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        compiled_namespace,
    )
    return compiled_namespace[method_name]


def _manager():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.rank = 0
    manager.gpu_blocks = 2
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=lambda: (_ for _ in ()).throw(
            AssertionError("event allocated in off mode")
        ),
        stream_id=id,
    )
    manager.logical_blocks = 4
    manager.logical_to_slot = {}
    manager.slot_to_logical = [None, None]
    manager.slot_last_used = [0, 0]
    manager.clock = 0
    manager.cpu_valid = [False] * 4
    manager.dirty_logical_blocks = set()
    manager.pending_wait_blocks = set()
    manager.bound_generations = [None] * 4
    manager.h2d_done = {}
    manager.d2h_done = {}
    manager.stats = {
        "evictions": 0,
        "evict_clean": 0,
    }
    return manager


def test_same_generation_binding_is_idempotent():
    manager = _manager()

    manager.bind_logical_block_identity(1, 3)
    manager.bind_logical_block_identity(1, 3)

    assert manager.bound_generations[1] == 3


def test_newer_generation_binding_clears_stale_metadata():
    manager = _manager()
    manager.bound_generations[1] = 3
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.cpu_valid[1] = True
    manager.dirty_logical_blocks.add(1)
    manager.pending_wait_blocks.add(1)
    manager.h2d_done[1] = object()
    manager.d2h_done[1] = object()

    manager.bind_logical_block_identity(1, 4)

    assert manager.bound_generations[1] == 4
    assert manager.logical_to_slot == {}
    assert manager.slot_to_logical == [None, None]
    assert manager.cpu_valid[1] is False
    assert manager.dirty_logical_blocks == set()
    assert manager.pending_wait_blocks == set()
    assert manager.h2d_done == {}
    assert manager.d2h_done == {}


def test_older_generation_binding_fails_closed():
    manager = _manager()
    manager.bound_generations[1] = 4

    with pytest.raises(RuntimeError, match="moved backwards"):
        manager.bind_logical_block_identity(1, 3)


def test_unbound_block_with_existing_bytes_fails_closed():
    manager = _manager()
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1

    with pytest.raises(RuntimeError, match="existing state"):
        manager.bind_logical_block_identity(1, 1)


def test_discard_generation_mismatch_is_atomic():
    manager = _manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1

    with pytest.raises(RuntimeError, match="generation mismatch"):
        manager.discard_resident_blocks(
            ((1, 6),),
            allow_dirty=False,
        )

    assert manager.logical_to_slot == {1: 0}
    assert manager.slot_to_logical == [1, None]


def test_discard_dirty_block_requires_explicit_permission():
    manager = _manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.dirty_logical_blocks.add(1)

    with pytest.raises(RuntimeError, match="dirty"):
        manager.discard_resident_blocks(
            ((1, 7),),
            allow_dirty=False,
        )


def test_discard_clears_block_local_metadata_without_unbinding_identity():
    manager = _manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.pending_wait_blocks.add(1)
    manager.h2d_done[1] = object()
    manager.d2h_done[1] = object()

    discarded = manager.discard_resident_blocks(
        ((1, 7),),
        allow_dirty=False,
    )

    assert discarded == ((1, 7),)
    assert manager.logical_to_slot == {}
    assert manager.slot_to_logical == [None, None]
    assert manager.pending_wait_blocks == set()
    assert manager.h2d_done == {}
    assert manager.d2h_done == {}
    assert manager.bound_generations[1] == 7


def test_discard_restores_snapshot_after_mid_batch_failure(monkeypatch):
    manager = _manager()
    for block_id, slot in ((1, 0), (2, 1)):
        manager.bound_generations[block_id] = 7
        manager.logical_to_slot[block_id] = slot
        manager.slot_to_logical[slot] = block_id
    original = manager._discard_validated_resident_block
    calls = 0

    def fail_second(block_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected discard failure")
        original(block_id)

    monkeypatch.setattr(
        manager,
        "_discard_validated_resident_block",
        fail_second,
    )

    with pytest.raises(
        RuntimeError,
        match="injected discard failure",
    ):
        manager.discard_resident_blocks(
            ((1, 7), (2, 7)),
            allow_dirty=False,
        )

    assert manager.logical_to_slot == {1: 0, 2: 1}
    assert manager.slot_to_logical == [1, 2]


def _clean_resident_manager():
    manager = _manager()
    for block_id, slot in ((1, 0), (2, 1)):
        manager.bound_generations[block_id] = 7
        manager.logical_to_slot[block_id] = slot
        manager.slot_to_logical[slot] = block_id
        manager.cpu_valid[block_id] = True
        manager.d2h_done[block_id] = object()
    return manager


def test_evict_clean_resident_blocks_preserves_cpu_generation():
    manager = _clean_resident_manager()

    evicted = manager.evict_clean_resident_blocks(
        ((2, 7), (1, 7)),
    )

    assert evicted == ((2, 7), (1, 7))
    assert manager.logical_to_slot == {}
    assert manager.slot_to_logical == [None, None]
    assert manager.cpu_valid[1] is True
    assert manager.cpu_valid[2] is True
    assert manager.bound_generations[1] == 7
    assert manager.bound_generations[2] == 7
    assert manager.d2h_done == {}
    assert manager.stats["evictions"] == 2
    assert manager.stats["evict_clean"] == 2


@pytest.mark.parametrize(
    "mutate,identities,match",
    [
        (
            lambda manager: manager.dirty_logical_blocks.add(2),
            ((1, 7), (2, 7)),
            "clean",
        ),
        (
            lambda manager: None,
            ((1, 7), (2, 6)),
            "generation mismatch",
        ),
        (
            lambda manager: manager.cpu_valid.__setitem__(2, False),
            ((1, 7), (2, 7)),
            "CPU-valid",
        ),
        (
            lambda manager: manager.pending_wait_blocks.add(2),
            ((1, 7), (2, 7)),
            "pending H2D",
        ),
        (
            lambda manager: None,
            ((1, 7), (1, 7)),
            "unique",
        ),
    ],
)
def test_evict_clean_resident_blocks_rejects_batch_atomically(
    mutate,
    identities,
    match,
):
    manager = _clean_resident_manager()
    mutate(manager)
    before = (
        dict(manager.logical_to_slot),
        list(manager.slot_to_logical),
        list(manager.cpu_valid),
        list(manager.bound_generations),
        dict(manager.d2h_done),
        dict(manager.stats),
    )

    with pytest.raises((ValueError, RuntimeError), match=match):
        manager.evict_clean_resident_blocks(identities)

    after = (
        dict(manager.logical_to_slot),
        list(manager.slot_to_logical),
        list(manager.cpu_valid),
        list(manager.bound_generations),
        dict(manager.d2h_done),
        dict(manager.stats),
    )
    assert after == before


def test_speculative_residency_summary_reads_real_stats():
    manager = _manager()
    manager.stats.update({
        "speculative_residency_prepares": 2,
        "speculative_residency_precommits": 1,
        "speculative_residency_seals": 1,
        "speculative_residency_rollbacks": 1,
        "speculative_residency_committed_blocks": 3,
        "speculative_residency_rejected_blocks": 4,
        "speculative_residency_rejected_d2h_copies": 0,
    })

    assert manager.speculative_residency_summary() == {
        "speculative_residency_prepares": 2,
        "speculative_residency_precommits": 1,
        "speculative_residency_seals": 1,
        "speculative_residency_rollbacks": 1,
        "speculative_residency_committed_blocks": 3,
        "speculative_residency_rejected_blocks": 4,
        "speculative_residency_rejected_d2h_copies": 0,
    }


def test_model_runner_reset_peak_memory_stats_synchronizes_and_snapshots():
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
            reset_peak_memory_stats=lambda: calls.append("reset"),
        )
    )
    method = _load_model_runner_method(
        "reset_peak_memory_stats",
        {"torch": fake_torch},
    )
    runner = SimpleNamespace(
        memory_snapshot=lambda: {
            "cuda_allocated_bytes": 7,
            "cuda_peak_allocated_bytes": 7,
        }
    )

    result = method(runner)

    assert calls == ["synchronize", "reset"]
    assert result["cuda_peak_allocated_bytes"] == 7
