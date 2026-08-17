"""Dependency-light tests for KVOffloadMVP0 diagnostic ownership."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import time
import types

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_MODEL_RUNNER_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "model_runner.py",
)
_DIAGNOSTIC_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "h2d_slot_reuse_diagnostic.py",
)


def _load_diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "h2d_slot_reuse_diagnostic_for_manager_test",
        _DIAGNOSTIC_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_kv_offload_class(diagnostic_module):
    source = open(_MODEL_RUNNER_PATH).read()
    tree = ast.parse(source, filename=_MODEL_RUNNER_PATH)
    future = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
    )
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KVOffloadMVP0"
    )
    selected = ast.Module(
        body=[future, target],
        type_ignores=[],
    )
    ast.fix_missing_locations(selected)
    module = types.ModuleType("kv_offload_manager_under_test")
    module.__dict__.update(
        H2DSlotReuseDiagnostic=(
            diagnostic_module.H2DSlotReuseDiagnostic
        ),
        time=time,
        torch=types.SimpleNamespace(),
    )
    exec(
        compile(selected, _MODEL_RUNNER_PATH, "exec"),
        module.__dict__,
    )
    return module.KVOffloadMVP0


_DIAGNOSTIC = _load_diagnostic_module()
KVOffloadMVP0 = _load_kv_offload_class(_DIAGNOSTIC)


class _DummyEvent:
    def __init__(self, ordinal, operation_log=None):
        self.ordinal = ordinal
        self.elapsed = {}
        self.operation_log = operation_log

    def record(self, stream=None):
        if self.operation_log is not None:
            self.operation_log.append(f"record:event_{self.ordinal}")

    def elapsed_time(self, other):
        return float(self.elapsed[other.ordinal])


class _EventFactory:
    def __init__(self, operation_log=None):
        self.next_ordinal = 1
        self.operation_log = operation_log

    def __call__(self):
        event = _DummyEvent(
            self.next_ordinal,
            self.operation_log,
        )
        self.next_ordinal += 1
        return event


class _DummyStream:
    def __init__(self, operation_log):
        self.operation_log = operation_log
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event.ordinal)
        self.operation_log.append(f"wait:event_{event.ordinal}")

    def wait_stream(self, stream):
        self.operation_log.append("wait:stream")


class _StreamContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _CopyView:
    def __init__(self, operation_log, label):
        self.operation_log = operation_log
        self.label = label

    def copy_(self, source, non_blocking):
        assert non_blocking is True
        self.operation_log.append(
            f"copy:{source.label}->{self.label}"
        )


class _CopyCache:
    def __init__(self, operation_log, name):
        self.operation_log = operation_log
        self.name = name

    def __getitem__(self, key):
        span = key[2]
        return _CopyView(
            self.operation_log,
            f"{self.name}[{span.start}:{span.stop}]",
        )


class _NoopKVOffload(KVOffloadMVP0):
    def __init__(self, *, gpu_blocks=2, logical_blocks=4):
        self.rank = 0
        self.gpu_blocks = int(gpu_blocks)
        self.logical_blocks = int(logical_blocks)
        self.block_nbytes = 1
        self.async_copy = True
        self.batch_copy = True
        self.writeback_on_evict = False
        self.evict_policy = "lru"
        self.logical_to_slot = {}
        self.slot_to_logical = [None] * self.gpu_blocks
        self.slot_last_used = [0] * self.gpu_blocks
        self.clock = 0
        self.cpu_valid = [False] * self.logical_blocks
        self.bound_generations = [0] * self.logical_blocks
        self.dirty_logical_blocks = set()
        self.pending_wait_blocks = set()
        self.h2d_done = {}
        self.d2h_done = {}
        self.copy_stream = None
        self.stats = {
            "h2d_copies": 0,
            "d2h_copies": 0,
            "evictions": 0,
            "h2d_ms": 0.0,
            "d2h_ms": 0.0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "evict_clean": 0,
            "evict_dirty": 0,
            "copy_waits": 0,
            "h2d_batches": 0,
            "d2h_batches": 0,
            "h2d_batch_spans": 0,
            "d2h_batch_spans": 0,
            "peak_resident_blocks": 0,
        }
        self.event_factory = _EventFactory()
        self._initialize_h2d_slot_reuse_diagnostic(
            event_factory=self.event_factory,
            stream_id=id,
        )

    def _enqueue_d2h_pairs(self, pairs):
        return None

    def _enqueue_h2d_pairs(self, pairs):
        return None

    def wait_for_blocks(self, logical_blocks, clear_pending=False):
        return None


def _active_mapping(manager):
    return tuple(
        (
            occupancy.physical_slot,
            occupancy.logical_block,
            occupancy.bound_generation,
            occupancy.occupancy_generation,
        )
        for occupancy in (
            manager._h2d_slot_reuse_diagnostic.active_occupancies
        )
    )


def test_h2d_slot_reuse_diagnostic_defaults_off_without_events():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.rank = 0
    manager.gpu_blocks = 2
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=lambda: (_ for _ in ()).throw(
            AssertionError("event allocated in off mode")
        ),
        stream_id=id,
    )
    assert manager.h2d_slot_reuse_diagnostic_summary() == {
        "rank": 0,
        "mode": "off",
        "retained_event_count": 0,
        "read_row_count": 0,
        "overwrite_row_count": 0,
    }


def test_h2d_slot_reuse_diagnostic_lifecycle_is_explicit():
    manager = _NoopKVOffload()
    assert manager.configure_h2d_slot_reuse_diagnostic(
        "observe"
    ) == {"rank": 0, "mode": "observe"}
    manager.configure_h2d_slot_reuse_diagnostic("off")
    assert manager.h2d_slot_reuse_diagnostic_summary()[
        "mode"
    ] == "off"


def test_read_wrapper_off_mode_does_not_request_cuda_stream():
    manager = _NoopKVOffload()
    assert manager.record_h2d_slot_read_window(
        engine_step=None,
        attention_stage="decode",
        layer_index=2,
        window_ordinal=0,
        logical_blocks=(0,),
        physical_slots=(0,),
        current_stream=None,
    ) is None


def test_initial_assignment_updates_diagnostic_occupancy():
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    assert manager.ensure_resident(
        [0],
        require_valid=False,
    ) == {0: 0}
    assert _active_mapping(manager) == ((0, 0, 0, 1),)


def test_reassignment_advances_physical_slot_generation():
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0], require_valid=False)
    first = _active_mapping(manager)[0][-1]
    manager.ensure_resident([1], require_valid=False)
    second = _active_mapping(manager)[0][-1]
    manager.ensure_resident([0], require_valid=False)
    third = _active_mapping(manager)[0][-1]
    assert (first, second, third) == (1, 2, 3)
    assert _active_mapping(manager)[:1] == ((0, 0, 0, 3),)


def test_clear_logical_block_metadata_releases_slot():
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0], require_valid=False)
    manager._clear_logical_block_metadata(0)
    assert manager.logical_to_slot == {}
    assert manager.slot_to_logical == [None]
    assert _active_mapping(manager) == ()


def test_mapping_assertion_rejects_production_drift():
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0], require_valid=False)
    manager.slot_to_logical[0] = 1
    with pytest.raises(RuntimeError, match="mapping"):
        manager._assert_h2d_slot_reuse_diagnostic_mapping()


def test_discard_resident_blocks_releases_only_requested_slots():
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=3)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0, 1], require_valid=False)
    assert manager.discard_resident_blocks(
        ((0, 0),),
        allow_dirty=True,
    ) == ((0, 0),)
    assert manager.logical_to_slot == {1: 1}
    assert _active_mapping(manager) == ((1, 1, 0, 1),)


def test_evict_clean_resident_blocks_releases_slots():
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=3)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0, 1], require_valid=False)
    manager.cpu_valid[0] = True
    assert manager.evict_clean_resident_blocks(
        ((0, 0),)
    ) == ((0, 0),)
    assert manager.logical_to_slot == {1: 1}
    assert _active_mapping(manager) == ((1, 1, 0, 1),)


def test_contiguous_slot_reorder_updates_final_diagnostic_mapping():
    manager = _NoopKVOffload(gpu_blocks=4, logical_blocks=8)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0, 1, 2, 3], require_valid=False)
    manager.slot_last_used[:] = [40, 20, 30, 10]
    mapping = manager.ensure_resident(
        [4, 5, 6, 7],
        require_valid=False,
    )
    assert mapping == {4: 0, 5: 1, 6: 2, 7: 3}
    assert tuple(
        (slot, logical_block)
        for slot, logical_block, _, _ in _active_mapping(manager)
    ) == ((0, 4), (1, 5), (2, 6), (3, 7))


def test_discard_failure_restores_diagnostic_mapping_atomically():
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=3)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0, 1], require_valid=False)
    before = _active_mapping(manager)
    original = manager._discard_validated_resident_block
    calls = 0

    def fail_after_first(logical_block):
        nonlocal calls
        calls += 1
        original(logical_block)
        if calls == 1:
            raise RuntimeError("injected discard failure")

    manager._discard_validated_resident_block = fail_after_first
    with pytest.raises(RuntimeError, match="injected discard failure"):
        manager.discard_resident_blocks(
            ((0, 0), (1, 0)),
            allow_dirty=True,
        )
    assert manager.logical_to_slot == {0: 0, 1: 1}
    assert manager.slot_to_logical == [0, 1]
    assert _active_mapping(manager) == before
    manager._assert_h2d_slot_reuse_diagnostic_mapping()


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (
            "observe",
            [
                "record:event_2",
                "copy:cpu[4:6]->gpu[0:2]",
                "record:event_3",
                "record:production",
            ],
        ),
        (
            "control",
            [
                "wait:event_1",
                "record:event_2",
                "copy:cpu[4:6]->gpu[0:2]",
                "record:event_3",
                "record:production",
            ],
        ),
    ),
)
def test_h2d_span_instrumentation_preserves_exact_order(mode, expected):
    operation_log = []
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=6)
    manager.event_factory = _EventFactory(operation_log)
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=manager.event_factory,
        stream_id=id,
    )
    manager.copy_stream = _DummyStream(operation_log)
    manager.kv_cache = _CopyCache(operation_log, "gpu")
    manager.cpu_cache = _CopyCache(operation_log, "cpu")
    manager._record_copy_event = lambda: (
        operation_log.append("record:production")
        or _DummyEvent(99)
    )
    manager.configure_h2d_slot_reuse_diagnostic(mode)
    manager.ensure_resident([0, 1], require_valid=False)
    manager.set_h2d_slot_reuse_context(
        engine_step=3,
        attention_stage="decode",
        layer_index=2,
        window_ordinal=0,
    )
    manager.record_h2d_slot_read_window(
        engine_step=None,
        attention_stage="decode",
        layer_index=2,
        window_ordinal=0,
        logical_blocks=(0, 1),
        physical_slots=(0, 1),
        current_stream=_DummyStream(operation_log),
    )
    operation_log.clear()
    manager.cpu_valid[4] = True
    manager.cpu_valid[5] = True
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            stream=lambda stream: _StreamContext(),
            current_stream=lambda: (_ for _ in ()).throw(
                AssertionError("current stream wait added")
            ),
        )
    )
    KVOffloadMVP0._enqueue_h2d_pairs.__globals__["torch"] = (
        fake_torch
    )

    manager.ensure_resident([4, 5], require_valid=True)
    KVOffloadMVP0._enqueue_h2d_pairs(
        manager,
        [(4, 0), (5, 1)],
    )

    assert operation_log == expected
    assert manager.copy_stream.waited == (
        [] if mode == "observe" else [1]
    )
    assert manager.stats["h2d_copies"] == 2
    assert manager.stats["h2d_batches"] == 1
    assert manager.stats["h2d_batch_spans"] == 1
    assert manager.h2d_done[4] is manager.h2d_done[5]


def test_enabled_diagnostic_rejects_synchronous_h2d_copy():
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.ensure_resident([0], require_valid=False)
    manager.cpu_valid[1] = True
    manager.copy_stream = None
    manager.kv_cache = _CopyCache([], "gpu")
    manager.cpu_cache = _CopyCache([], "cpu")
    KVOffloadMVP0._enqueue_h2d_pairs.__globals__["torch"] = (
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                synchronize=lambda: None,
            )
        )
    )
    with pytest.raises(
        RuntimeError,
        match="requires asynchronous copy stream",
    ):
        manager.ensure_resident([1], require_valid=True)
        KVOffloadMVP0._enqueue_h2d_pairs(manager, [(1, 0)])


def _run_inventory_mode(mode):
    operation_log = []
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=6)
    manager.event_factory = _EventFactory(operation_log)
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=manager.event_factory,
        stream_id=id,
    )
    manager.copy_stream = _DummyStream(operation_log)
    manager.kv_cache = _CopyCache(operation_log, "gpu")
    manager.cpu_cache = _CopyCache(operation_log, "cpu")
    production_event = _DummyEvent(99)
    manager._record_copy_event = lambda: production_event
    manager.configure_h2d_slot_reuse_diagnostic(mode)
    if mode != "off":
        manager.set_h2d_slot_reuse_context(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=0,
        )
    manager.ensure_resident([0, 1], require_valid=False)
    manager.cpu_valid[4] = True
    manager.cpu_valid[5] = True
    KVOffloadMVP0._enqueue_h2d_pairs.__globals__["torch"] = (
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                stream=lambda stream: _StreamContext(),
            )
        )
    )
    manager.ensure_resident([4, 5], require_valid=True)
    KVOffloadMVP0._enqueue_h2d_pairs(
        manager,
        [(4, 0), (5, 1)],
    )
    return {
        "spans": manager._coalesce_copy_pairs(
            [(4, 0), (5, 1)]
        ),
        "h2d_copies": manager.stats["h2d_copies"],
        "h2d_bytes": manager.stats["h2d_bytes"],
        "h2d_batches": manager.stats["h2d_batches"],
        "h2d_batch_spans": manager.stats[
            "h2d_batch_spans"
        ],
        "pending_wait_blocks": set(
            manager.pending_wait_blocks
        ),
        "h2d_keys": set(manager.h2d_done),
        "shared_event": (
            manager.h2d_done[4] is manager.h2d_done[5]
        ),
        "d2h_copies": manager.stats["d2h_copies"],
        "d2h_bytes": manager.stats["d2h_bytes"],
    }


def test_diagnostic_modes_preserve_h2d_and_d2h_inventory():
    off = _run_inventory_mode("off")
    observe = _run_inventory_mode("observe")
    control = _run_inventory_mode("control")
    assert observe == off
    assert control == off


def test_enabled_diagnostic_summary_records_exact_copy_inventories():
    operation_log = []
    manager = _NoopKVOffload(gpu_blocks=2, logical_blocks=6)
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=_EventFactory(operation_log),
        stream_id=id,
    )
    manager.copy_stream = _DummyStream(operation_log)
    manager.kv_cache = _CopyCache(operation_log, "gpu")
    manager.cpu_cache = _CopyCache(operation_log, "cpu")
    manager._record_copy_event = lambda: _DummyEvent(99)
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.set_h2d_slot_reuse_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    manager.ensure_resident([0, 1], require_valid=False)
    manager.cpu_valid[4] = True
    manager.cpu_valid[5] = True
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            stream=lambda stream: _StreamContext(),
            current_stream=lambda: _DummyStream(operation_log),
        )
    )
    KVOffloadMVP0._enqueue_h2d_pairs.__globals__["torch"] = (
        fake_torch
    )
    KVOffloadMVP0._enqueue_d2h_pairs.__globals__["torch"] = (
        fake_torch
    )

    manager.ensure_resident([4, 5], require_valid=True)
    KVOffloadMVP0._enqueue_h2d_pairs(
        manager,
        [(4, 0), (5, 1)],
    )
    KVOffloadMVP0._enqueue_d2h_pairs(
        manager,
        [(0, 0), (1, 1)],
    )

    summary = manager.summary()
    assert summary["h2d_pair_inventory"] == [[4, 0], [5, 1]]
    assert summary["h2d_span_inventory"] == [[4, 0, 2]]
    assert summary["d2h_pair_inventory"] == [[0, 0], [1, 1]]
    assert summary["d2h_span_inventory"] == [[0, 0, 2]]


@pytest.mark.parametrize(
    ("seed_old", "expected_status"),
    (
        (False, "NO_PRIOR_OCCUPANCY"),
        (True, "NO_PRIOR_READ"),
    ),
)
def test_manager_drain_reports_explicit_no_prior_status(
    seed_old,
    expected_status,
):
    manager = _NoopKVOffload(gpu_blocks=1, logical_blocks=2)
    manager.event_factory = _EventFactory()
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=manager.event_factory,
        stream_id=id,
    )
    manager.copy_stream = _DummyStream([])
    manager.kv_cache = _CopyCache([], "gpu")
    manager.cpu_cache = _CopyCache([], "cpu")
    manager._record_copy_event = lambda: _DummyEvent(99)
    manager.synchronize_copies = lambda: None
    manager.configure_h2d_slot_reuse_diagnostic("observe")
    manager.set_h2d_slot_reuse_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    if seed_old:
        manager.ensure_resident([0], require_valid=False)
    manager.cpu_valid[1] = True
    KVOffloadMVP0._enqueue_h2d_pairs.__globals__["torch"] = (
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                stream=lambda stream: _StreamContext(),
            )
        )
    )
    manager.ensure_resident([1], require_valid=True)
    KVOffloadMVP0._enqueue_h2d_pairs(manager, [(1, 0)])
    artifact = manager.drain_h2d_slot_reuse_diagnostic(
        timing_epsilon_ms=0.2
    )
    assert artifact["overwrite_rows"][0][
        "timing_status"
    ] == expected_status
