from __future__ import annotations

import ast
import hashlib
from itertools import count
import importlib.util
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm" / "engine" / "model_runner.py"
FAKE_TORCH = SimpleNamespace(
    int32=object(),
    int64=object(),
)


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [str(ROOT / "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [str(ROOT / "tinyvllm" / "engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(
            self._hash.digest(),
            "little",
        )


xxhash_mod = types.ModuleType("xxhash")
xxhash_mod.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_mod)

sampling_params_mod = _load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_mod = _load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
block_manager_mod = _load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
speculative_residency_mod = _load_module(
    "tinyvllm.engine.speculative_residency",
    "tinyvllm/engine/speculative_residency.py",
)
h2d_slot_reuse_diagnostic_mod = _load_module(
    "tinyvllm.engine.h2d_slot_reuse_diagnostic",
    "tinyvllm/engine/h2d_slot_reuse_diagnostic.py",
)

SamplingParams = sampling_params_mod.SamplingParams
Sequence = sequence_mod.Sequence
BlockManager = block_manager_mod.BlockManager
build_kv_block_identity_rows = (
    speculative_residency_mod.build_kv_block_identity_rows
)


def _load_kv_offload_type():
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    )
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
        "H2DSlotReuseDiagnostic": (
            h2d_slot_reuse_diagnostic_mod.H2DSlotReuseDiagnostic
        ),
        "time": time,
        "torch": FAKE_TORCH,
    }
    exec(
        compile(module, str(MODEL_RUNNER_PATH), "exec"),
        namespace,
    )
    return namespace["KVOffloadMVP0"]


def _load_prepare_prefill(set_context):
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
        and node.name == "prepare_prefill"
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {
        "torch": FAKE_TORCH,
        "set_context": set_context,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["prepare_prefill"]


KVOffloadMVP0 = _load_kv_offload_type()


def _sequence(token_ids):
    return Sequence(
        list(token_ids),
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=False,
        ),
    )


def _publish_and_release_prefix(
    block_manager: BlockManager,
    token_ids,
):
    sequence = _sequence(token_ids)
    block_manager.allocate(
        sequence,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(
        sequence,
        0,
        len(sequence),
    )
    block_ids = tuple(sequence.block_table)
    block_manager.deallocate(sequence)
    return block_ids


def _offload_manager(logical_blocks: int, gpu_blocks: int = 1):
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.rank = 0
    manager.logical_blocks = logical_blocks
    manager.gpu_blocks = gpu_blocks
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=lambda: (_ for _ in ()).throw(
            AssertionError("event allocated in off mode")
        ),
        stream_id=id,
    )
    manager.logical_to_slot = {}
    manager.slot_to_logical = [None] * gpu_blocks
    manager.slot_last_used = [0] * gpu_blocks
    manager.clock = 0
    manager.cpu_valid = [False] * logical_blocks
    manager.dirty_logical_blocks = set()
    manager.pending_wait_blocks = set()
    manager.bound_generations = [None] * logical_blocks
    manager.h2d_done = {}
    manager.d2h_done = {}
    manager.evict_policy = "lru"
    manager.block_nbytes = 1
    manager.stats = {
        "evictions": 0,
        "evict_clean": 0,
        "evict_dirty": 0,
        "copy_waits": 0,
    }
    manager.copy_stream = None
    manager.h2d_pairs = []
    manager.d2h_pairs = []
    manager._enqueue_h2d_pairs = (
        lambda pairs: manager.h2d_pairs.extend(pairs)
    )
    manager._enqueue_d2h_pairs = (
        lambda pairs: manager.d2h_pairs.extend(pairs)
    )
    return manager


def test_cpu_backed_shared_prefix_survives_idle_release_and_reuse():
    Sequence.block_size = 4
    Sequence.counter = count()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    prefix_block, _ = _publish_and_release_prefix(
        block_manager,
        [1, 2, 3, 4, 5],
    )
    prefix_generation = block_manager.blocks[
        prefix_block
    ].generation
    offload = _offload_manager(logical_blocks=4)
    offload.bind_logical_block_identity(
        prefix_block,
        prefix_generation,
    )
    offload.cpu_valid[prefix_block] = True

    reused = _sequence([1, 2, 3, 4, 9])
    reservation = block_manager.reserve_sequence_blocks(
        reused,
        max_cached_tokens=4,
    )
    block_manager.attach_sequence_reservation(
        reservation,
        reused,
    )
    identity_rows = build_kv_block_identity_rows(
        block_manager,
        (reused,),
    )
    for block_id, generation in identity_rows[0].block_identities:
        offload.bind_logical_block_identity(
            block_id,
            generation,
        )

    mapping = offload.ensure_resident(
        [prefix_block],
        require_valid=True,
    )

    assert reservation.prefix_block_count == 1
    assert reused.block_table[0] == prefix_block
    assert block_manager.blocks[prefix_block].generation == (
        prefix_generation
    )
    assert offload.cpu_valid[prefix_block] is True
    assert mapping == {prefix_block: 0}
    assert offload.h2d_pairs == [(prefix_block, 0)]
    assert offload.d2h_pairs == []
    block_manager.deallocate(reused)


def test_recycled_prefix_generation_invalidates_old_cpu_backing():
    Sequence.block_size = 4
    Sequence.counter = count()
    block_manager = BlockManager(num_blocks=1, block_size=4)
    (block_id,) = _publish_and_release_prefix(
        block_manager,
        [1, 2, 3, 4],
    )
    old_generation = block_manager.blocks[block_id].generation
    offload = _offload_manager(logical_blocks=1)
    offload.bind_logical_block_identity(
        block_id,
        old_generation,
    )
    offload.cpu_valid[block_id] = True
    block_manager.clear_reusable_cache()

    replacement = _sequence([9, 10, 11, 12])
    block_manager.allocate(
        replacement,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    identity_rows = build_kv_block_identity_rows(
        block_manager,
        (replacement,),
    )
    new_generation = identity_rows[0].block_identities[0][1]
    offload.bind_logical_block_identity(
        block_id,
        new_generation,
    )

    assert new_generation == old_generation + 1
    assert offload.cpu_valid[block_id] is False
    with pytest.raises(
        RuntimeError,
        match="requested unreadable logical block",
    ):
        offload.ensure_resident(
            [block_id],
            require_valid=True,
        )
    block_manager.deallocate(replacement)


class _DefaultConfig(SimpleNamespace):
    def __getattr__(self, name):
        del name
        return 0


class _RecordingOffload:
    def __init__(self):
        self.stats = {
            "prefetch_plans": 0,
            "prefetch_write_blocks": 0,
        }
        self.slot_calls = []
        self.row_calls = []

    def translate_slots_for_positions(
        self,
        block_table,
        positions,
        require_valid=False,
        future_logical_blocks=None,
    ):
        self.slot_calls.append(
            {
                "block_table": tuple(block_table),
                "positions": tuple(positions),
                "require_valid": bool(require_valid),
                "future_logical_blocks": frozenset(
                    future_logical_blocks or ()
                ),
            }
        )
        return list(positions)

    def translate_block_rows(
        self,
        rows,
        require_valid=True,
        future_logical_blocks=None,
    ):
        self.row_calls.append(
            {
                "rows": tuple(tuple(row) for row in rows),
                "require_valid": bool(require_valid),
                "future_logical_blocks": frozenset(
                    future_logical_blocks or ()
                ),
            }
        )
        return rows


def test_prepare_prefill_requires_valid_cpu_backing_for_cached_prefix():
    Sequence.block_size = 4
    Sequence.counter = count()
    context_calls = []
    prepare_prefill = _load_prepare_prefill(
        lambda *args, **kwargs: context_calls.append(
            (args, kwargs)
        )
    )
    offload = _RecordingOffload()
    runner = SimpleNamespace(
        kv_offload=offload,
        block_size=4,
        config=_DefaultConfig(
            kv_offload_blockwise_prefill=False,
            am_compact_blocks=0,
        ),
        _kv_offload_pending_dirty_blocks=[],
    )
    runner._kv_offload_translate_slots_for_positions = (
        lambda block_table, positions, require_valid=False,
        future_logical_blocks=None:
        offload.translate_slots_for_positions(
            block_table,
            positions,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
    )
    runner._kv_offload_translate_block_rows = (
        lambda rows, require_valid=True,
        future_logical_blocks=None:
        offload.translate_block_rows(
            rows,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
    )
    runner._kv_offload_mark_pending_dirty = (
        lambda block_table, positions: None
    )
    runner._list_to_cuda = (
        lambda data, name, dtype: list(data)
    )
    runner.prepare_block_tables_from_rows = (
        lambda rows: tuple(tuple(row) for row in rows)
    )
    sequence = _sequence([1, 2, 3, 4, 5])
    sequence.block_table = [2, 3]
    sequence.num_cached_tokens = 4
    sequence.num_computed_tokens = 4
    sequence.prefill_chunk_start = 4
    sequence.prefill_chunk_end = 5

    input_ids, positions = prepare_prefill(
        runner,
        [sequence],
    )

    prefix_call = next(
        call
        for call in offload.slot_calls
        if call["positions"] == (0, 1, 2, 3)
    )
    write_call = next(
        call
        for call in offload.slot_calls
        if call["positions"] == (4,)
    )
    assert input_ids == [5]
    assert positions == [4]
    assert prefix_call["require_valid"] is True
    assert write_call["require_valid"] is False
    assert offload.row_calls == [{
        "rows": ((2, 3),),
        "require_valid": False,
        "future_logical_blocks": frozenset({2, 3}),
    }]
    assert len(context_calls) == 1
