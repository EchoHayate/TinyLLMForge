"""Dependency-light tests for spec-verify CUDA Graph admission."""

from __future__ import annotations

import ast
import inspect
import math
import os
import sys
from types import SimpleNamespace

import pytest

from tools import test_model_runner_spec_verify as base


model_runner = base.model_runner
_SPEC_VERIFY_CACHE_MODULE = sys.modules[
    "tinyvllm.engine.spec_verify_exact_cuda_graph_cache"
]


class ShapeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype,
        element_size,
        device="cuda:0",
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._element_size = element_size
        self.device = device

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self._element_size


class RuntimeTensor(ShapeTensor):
    def __init__(
        self,
        shape,
        *,
        dtype,
        element_size,
        device="cuda:0",
        data=None,
        value=None,
    ):
        super().__init__(
            shape,
            dtype=dtype,
            element_size=element_size,
            device=device,
        )
        self.data = data
        self.value = value
        self.copied_from = None

    def copy_(self, value):
        self.copied_from = value
        self.data = getattr(value, "data", self.data)
        self.value = getattr(value, "value", self.value)
        return self

    def tolist(self):
        return self.data


def make_context(*, batch_size=4, query_len=3, width=5, mode="spec_verify"):
    total = batch_size * query_len
    return SimpleNamespace(
        mode=mode,
        spec_verify_query_lens=(query_len,) * batch_size,
        slot_mapping=ShapeTensor(
            (total,),
            dtype="torch.int32",
            element_size=4,
        ),
        context_lens=ShapeTensor(
            (batch_size,),
            dtype="torch.int32",
            element_size=4,
        ),
        block_tables=ShapeTensor(
            (batch_size, width),
            dtype="torch.int32",
            element_size=4,
        ),
        flash_attn_num_splits=16,
    )


def make_runner(**overrides):
    defaults = {
        "spec_verify_cuda_graphs": True,
        "spec_verify_cuda_graph_batch_allowlist": (1, 4),
        "spec_verify_cuda_graph_query_len_allowlist": (1, 3),
    }
    defaults.update(overrides)
    runner = base.make_runner(**defaults)
    runner.world_size = 1
    runner.hybrid_state_runtime_bridge = None
    return runner


def make_input(batch_size=4, query_len=3):
    return ShapeTensor(
        (batch_size * query_len,),
        dtype="torch.int64",
        element_size=8,
    )


@pytest.mark.parametrize(
    ("runner_overrides", "context_overrides", "call_overrides", "expected"),
    (
        (
            {"spec_verify_cuda_graphs": False},
            {},
            {},
            ("feature_disabled", True),
        ),
        ({}, {}, {"enforce_eager": True}, ("enforce_eager", True)),
        ({}, {"mode": "decode"}, {}, ("unsupported_mode", True)),
        ({}, {}, {"world_size": 2}, ("tp_not_one", True)),
        (
            {"kv_offload_mvp0": True},
            {},
            {},
            ("kv_offload_enabled", True),
        ),
        (
            {"kv_offload_blockwise_decode": True},
            {},
            {},
            ("blockwise_enabled", True),
        ),
        (
            {"spec_verify_cuda_graph_batch_allowlist": (1,)},
            {},
            {},
            ("batch_not_allowlisted", True),
        ),
        (
            {"spec_verify_cuda_graph_query_len_allowlist": (1,)},
            {},
            {},
            ("query_len_not_allowlisted", True),
        ),
        ({}, {}, {"input_embeds": object()}, ("input_embeds_active", True)),
        (
            {},
            {},
            {"return_hidden": True},
            ("hidden_state_return_active", True),
        ),
        (
            {},
            {},
            {"hybrid_state_runtime_bridge": object()},
            ("non_transactional_state", False),
        ),
        (
            {},
            {},
            {"transaction_authorized": False},
            ("transaction_unauthorized", False),
        ),
    ),
)
def test_spec_verify_cuda_graph_admission_reasons_are_exact(
    runner_overrides,
    context_overrides,
    call_overrides,
    expected,
):
    runner = make_runner(**runner_overrides)
    runner.enforce_eager = call_overrides.get(
        "enforce_eager",
        False,
    )
    runner.world_size = call_overrides.get("world_size", 1)
    runner.hybrid_state_runtime_bridge = call_overrides.get(
        "hybrid_state_runtime_bridge",
        None,
    )
    context = make_context(**context_overrides)

    result = runner._spec_verify_graph_incompatible_reason(
        input_ids=make_input(),
        input_embeds=call_overrides.get("input_embeds"),
        return_hidden=call_overrides.get(
            "return_hidden",
            False,
        ),
        context=context,
        transaction_authorized=call_overrides.get(
            "transaction_authorized",
            True,
        ),
    )

    assert result == expected


@pytest.mark.parametrize(
    (
        "runner_overrides",
        "world_size",
        "expected_reason",
    ),
    (
        (
            {"spec_verify_cuda_graphs": False},
            1,
            "feature_disabled",
        ),
        ({}, 2, "tp_not_one"),
        (
            {"kv_offload_mvp0": True},
            1,
            "kv_offload_enabled",
        ),
        (
            {"kv_offload_blockwise_decode": True},
            1,
            "blockwise_enabled",
        ),
    ),
)
def test_spec_verify_incompatible_graph_modes_preserve_eager_output(
    runner_overrides,
    world_size,
    expected_reason,
):
    runner = make_runner(**runner_overrides)
    _install_spec_verify_runtime(runner)
    runner.world_size = world_size

    class Model:
        def __init__(self):
            self.forward_calls = 0
            self.compute_logits_calls = 0

        def __call__(
            self,
            input_ids,
            positions,
            input_embeds=None,
        ):
            del positions, input_embeds
            self.forward_calls += 1
            return RuntimeTensor(
                (input_ids.size(0), 8),
                dtype="torch.bfloat16",
                element_size=2,
                value="eager-output",
            )

        def compute_logits(self, outputs):
            self.compute_logits_calls += 1
            return f"logits:{outputs.value}"

    runner.model = Model()
    _set_spec_verify_runtime_context()
    try:
        result = runner.run_model(
            _runtime_input(),
            _runtime_positions(),
            False,
            execution_mode="spec_verify",
        )
    finally:
        base.context.reset_context()

    assert result == "logits:eager-output"
    assert runner.model.forward_calls == 1
    assert runner.model.compute_logits_calls == 1
    event = runner.spec_verify_graph_dispatch_observation()
    assert event["dispatch"] == "eager"
    assert event["fallback_reason"] == expected_reason


@pytest.mark.parametrize(
    ("batch_size", "query_len"),
    ((1, 1), (4, 3)),
)
def test_spec_verify_cuda_graph_admission_accepts_exact_allowlisted_family(
    batch_size,
    query_len,
):
    runner = make_runner()

    assert runner._spec_verify_graph_incompatible_reason(
        input_ids=make_input(batch_size, query_len),
        input_embeds=None,
        return_hidden=False,
        context=make_context(
            batch_size=batch_size,
            query_len=query_len,
        ),
        transaction_authorized=True,
    ) == (None, True)


def _install_identity_metadata(runner):
    runner.config.hf_config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    runner.block_size = 256
    runner.world_size = 1
    runner.kv_cache = SimpleNamespace(device="cuda:0")
    model_runner.flash_attn.__version__ = "2.7.4"
    model_runner.torch.cuda.get_device_capability = (
        lambda device: (8, 0)
    )


def test_spec_verify_graph_identity_uses_exact_tensor_shapes():
    runner = make_runner()
    _install_identity_metadata(runner)
    context = make_context()
    outputs = ShapeTensor(
        (12, 4096),
        dtype="torch.bfloat16",
        element_size=2,
    )

    identity = runner._build_spec_verify_graph_identity(
        input_ids=make_input(),
        outputs=outputs,
        context=context,
    )

    assert identity.active_batch_size == 4
    assert identity.query_len == 3
    assert identity.total_query_tokens == 12
    assert identity.page_table_width == 5
    assert identity.flash_attn_num_splits == 16
    assert identity.input_dtype == "torch.int64"
    assert identity.output_dtype == "torch.bfloat16"
    assert identity.num_query_heads == 32
    assert identity.num_kv_heads == 8
    assert identity.head_dim == 128
    assert identity.page_block_size == 256
    assert identity.device_compute_capability == (8, 0)


@pytest.mark.parametrize(
    ("field", "shape"),
    (
        ("input_ids", (11,)),
        ("slot_mapping", (11,)),
        ("context_lens", (3,)),
        ("block_tables", (3, 5)),
        ("outputs", (11, 4096)),
    ),
)
def test_spec_verify_graph_identity_rejects_shape_drift(field, shape):
    runner = make_runner()
    _install_identity_metadata(runner)
    context = make_context()
    input_ids = make_input()
    outputs = ShapeTensor(
        (12, 4096),
        dtype="torch.bfloat16",
        element_size=2,
    )
    if field == "input_ids":
        input_ids = ShapeTensor(
            shape,
            dtype="torch.int64",
            element_size=8,
        )
    elif field == "outputs":
        outputs = ShapeTensor(
            shape,
            dtype="torch.bfloat16",
            element_size=2,
        )
    else:
        original = getattr(context, field)
        setattr(
            context,
            field,
            ShapeTensor(
                shape,
                dtype=original.dtype,
                element_size=original.element_size(),
            ),
        )

    with pytest.raises(ValueError, match="shape"):
        runner._build_spec_verify_graph_identity(
            input_ids=input_ids,
            outputs=outputs,
            context=context,
        )


def test_spec_verify_graph_identity_rejects_unrepresented_output_width():
    runner = make_runner()
    _install_identity_metadata(runner)
    runner.config.hf_config.hidden_size = 3072

    with pytest.raises(ValueError, match="output width"):
        runner._build_spec_verify_graph_identity(
            input_ids=make_input(),
            outputs=ShapeTensor(
                (12, 3072),
                dtype="torch.bfloat16",
                element_size=2,
            ),
            context=make_context(),
        )


def test_spec_verify_graph_static_bytes_sum_exact_tensors():
    runner = make_runner()
    _install_identity_metadata(runner)
    context = make_context()
    input_ids = make_input()
    outputs = ShapeTensor(
        (12, 4096),
        dtype="torch.bfloat16",
        element_size=2,
    )
    identity = runner._build_spec_verify_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )

    assert runner._estimate_spec_verify_graph_static_bytes(
        identity,
    ) == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            input_ids,
            context.slot_mapping,
            context.context_lens,
            context.block_tables,
            outputs,
        )
    )


def test_spec_verify_dispatch_event_schema_is_fixed_and_independent():
    runner = make_runner()
    cache_module = _SPEC_VERIFY_CACHE_MODULE
    runner.spec_verify_exact_cuda_graph_cache = (
        cache_module.SpecVerifyExactCudaGraphCache(
            cache_module.SpecVerifyExactCudaGraphCacheConfig(
                enabled=True,
                batch_allowlist=(1, 4),
                query_len_allowlist=(1, 3),
                min_observations=2,
                max_entries=8,
                max_static_bytes=1024,
                max_reserved_bytes=2048,
                max_total_capture_ns=1000,
                max_single_capture_ns=500,
            )
        )
    )
    runner.last_cuda_graph_dispatch_event = {"decode": True}
    runner.last_spec_verify_cuda_graph_dispatch_event = None
    runner._spec_verify_cuda_graph_step_id = 0
    runner._spec_verify_cuda_graph_request_ids_hash = "request-hash"

    runner._publish_spec_verify_graph_dispatch_event(
        identity=None,
        dispatch="eager",
        decision="feature_disabled",
        fallback_reason="feature_disabled",
        cache_state="absent",
        observation_count=0,
        capture_attempted=False,
        capture_entry=None,
        transaction_authorized=False,
    )

    event = runner.spec_verify_graph_dispatch_observation()
    assert tuple(event) == model_runner.SPEC_VERIFY_DISPATCH_EVENT_FIELDS
    assert event["mode"] == "spec_verify"
    assert event["transaction_authorized"] is False
    assert runner.last_cuda_graph_dispatch_event == {"decode": True}


def test_model_runner_initializes_independent_spec_verify_graph_state():
    tree = ast.parse(
        open(base._MODEL_RUNNER_PATH, encoding="utf-8").read()
    )
    model_runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    init_function = next(
        node
        for node in model_runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )
    stored = {
        node.attr
        for statement in init_function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
    }
    assert {
        "spec_verify_exact_cuda_graph_cache",
        "last_spec_verify_cuda_graph_dispatch_event",
        "_spec_verify_cuda_graph_step_id",
    }.issubset(stored)


def _install_spec_verify_runtime(runner, *, min_observations=2):
    cache_module = _SPEC_VERIFY_CACHE_MODULE
    runner.spec_verify_exact_cuda_graph_cache = (
        cache_module.SpecVerifyExactCudaGraphCache(
            cache_module.SpecVerifyExactCudaGraphCacheConfig(
                enabled=True,
                batch_allowlist=(1, 4),
                query_len_allowlist=(1, 3),
                min_observations=min_observations,
                max_entries=8,
                max_static_bytes=64 * 1024 * 1024,
                max_reserved_bytes=512 * 1024 * 1024,
                max_total_capture_ns=5_000_000_000,
                max_single_capture_ns=2_000_000_000,
            )
        )
    )
    runner.config.hf_config = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_size=8,
        torch_dtype="torch.bfloat16",
    )
    runner.block_size = 4
    runner.kv_cache = SimpleNamespace(device="cuda:0")
    runner._spec_verify_transaction_authorized = True
    runner._spec_verify_cuda_graph_step_id = 0
    runner._spec_verify_cuda_graph_request_ids_hash = "request-hash"
    runner.last_spec_verify_cuda_graph_dispatch_event = None
    model_runner.flash_attn.__version__ = "2.7.4"
    model_runner.torch.cuda.get_device_capability = (
        lambda device: (8, 0)
    )
    return cache_module


def _set_spec_verify_runtime_context(
    *,
    batch_size=1,
    query_len=3,
    width=2,
):
    total = batch_size * query_len
    base.context.set_context(
        mode="spec_verify",
        slot_mapping=RuntimeTensor(
            (total,),
            dtype="torch.int32",
            element_size=4,
            data=list(range(total)),
        ),
        context_lens=RuntimeTensor(
            (batch_size,),
            dtype="torch.int32",
            element_size=4,
            data=[query_len] * batch_size,
        ),
        block_tables=RuntimeTensor(
            (batch_size, width),
            dtype="torch.int32",
            element_size=4,
            data=[
                list(range(width))
                for _ in range(batch_size)
            ],
        ),
        spec_verify_query_lens=(query_len,) * batch_size,
        flash_attn_num_splits=16,
    )


def _runtime_input(batch_size=1, query_len=3):
    total = batch_size * query_len
    return RuntimeTensor(
        (total,),
        dtype="torch.int64",
        element_size=8,
        data=list(range(total)),
    )


def _runtime_positions(batch_size=1, query_len=3):
    total = batch_size * query_len
    return RuntimeTensor(
        (total,),
        dtype="torch.int64",
        element_size=8,
        data=list(range(total)),
    )


def _install_ready_replay_entry(runner, cache_module, graph):
    context = SimpleNamespace(
        mode="spec_verify",
        spec_verify_query_lens=(3,),
        slot_mapping=RuntimeTensor(
            (3,),
            dtype="torch.int32",
            element_size=4,
            data=[11, 12, 13],
        ),
        context_lens=RuntimeTensor(
            (1,),
            dtype="torch.int32",
            element_size=4,
            data=[3],
        ),
        block_tables=RuntimeTensor(
            (1, 2),
            dtype="torch.int32",
            element_size=4,
            data=[[1, 2]],
        ),
        flash_attn_num_splits=16,
    )
    input_ids = _runtime_input()
    positions = _runtime_positions()
    outputs = RuntimeTensor(
        (3, 8),
        dtype="torch.bfloat16",
        element_size=2,
        value="graph-output",
    )
    identity = runner._build_spec_verify_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )
    entry = cache_module.SpecVerifyExactCudaGraphEntry(
        identity=identity,
        identity_sha256=identity.sha256,
        graph=graph,
        tensors={
            "input_ids": RuntimeTensor(
                (3,),
                dtype="torch.int64",
                element_size=8,
            ),
            "positions": RuntimeTensor(
                (3,),
                dtype="torch.int64",
                element_size=8,
            ),
            "slot_mapping": RuntimeTensor(
                (3,),
                dtype="torch.int32",
                element_size=4,
            ),
            "context_lens": RuntimeTensor(
                (1,),
                dtype="torch.int32",
                element_size=4,
            ),
            "block_tables": RuntimeTensor(
                (1, 2),
                dtype="torch.int32",
                element_size=4,
            ),
            "outputs": outputs,
        },
        static_bytes=256,
        capture_duration_ns=100,
        allocated_delta_bytes=0,
        reserved_delta_bytes=0,
    )
    runner.spec_verify_exact_cuda_graph_cache.capturing[
        identity.sha256
    ] = entry.static_bytes
    runner.spec_verify_exact_cuda_graph_cache.commit_capture(entry)
    return context, input_ids, positions, entry


def test_spec_verify_cold_capture_and_exact_replay_state_machine():
    runner = make_runner()
    cache_module = _install_spec_verify_runtime(runner)
    events = []

    class Model:
        def __init__(self):
            self.forward_calls = 0
            self.compute_logits_calls = 0

        def __call__(self, input_ids, positions, input_embeds=None):
            del positions, input_embeds
            self.forward_calls += 1
            events.append("live_eager")
            return RuntimeTensor(
                (input_ids.size(0), 8),
                dtype="torch.bfloat16",
                element_size=2,
                value=f"live-{self.forward_calls}",
            )

        def compute_logits(self, outputs):
            self.compute_logits_calls += 1
            return f"logits:{outputs.value}"

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    runner.model = Model()
    graph = Graph()

    def capture_after_live(
        *,
        identity,
        live_input_ids,
        live_positions,
        live_context,
    ):
        del live_input_ids, live_positions, live_context
        events.append("capture")
        entry = cache_module.SpecVerifyExactCudaGraphEntry(
            identity=identity,
            identity_sha256=identity.sha256,
            graph=graph,
            tensors={
                "outputs": RuntimeTensor(
                    (
                        identity.total_query_tokens,
                        identity.num_query_heads * identity.head_dim,
                    ),
                    dtype=identity.output_dtype,
                    element_size=2,
                    value="graph",
                )
            },
            static_bytes=runner._estimate_spec_verify_graph_static_bytes(
                identity
            ),
            capture_duration_ns=100,
            allocated_delta_bytes=0,
            reserved_delta_bytes=0,
        )
        runner.spec_verify_exact_cuda_graph_cache.commit_capture(entry)
        return entry

    def replay(entry, *, input_ids, positions, context):
        del input_ids, positions, context
        entry.graph.replay()
        return entry.tensors["outputs"]

    runner._attempt_post_step_spec_verify_capture = capture_after_live
    runner._replay_spec_verify_graph = replay
    _set_spec_verify_runtime_context()
    input_ids = _runtime_input()
    positions = _runtime_positions()

    first = runner.run_model(
        input_ids,
        positions,
        False,
        execution_mode="spec_verify",
    )
    second = runner.run_model(
        input_ids,
        positions,
        False,
        execution_mode="spec_verify",
    )
    third = runner.run_model(
        input_ids,
        positions,
        False,
        execution_mode="spec_verify",
    )

    assert first == "logits:live-1"
    assert second == "logits:live-2"
    assert third == "logits:graph"
    assert runner.model.forward_calls == 2
    assert runner.model.compute_logits_calls == 3
    assert graph.replays == 1
    assert events == ["live_eager", "live_eager", "capture"]

    _set_spec_verify_runtime_context(query_len=1)
    runner.run_model(
        _runtime_input(query_len=1),
        _runtime_positions(query_len=1),
        False,
        execution_mode="spec_verify",
    )
    _set_spec_verify_runtime_context(width=3)
    runner.run_model(
        input_ids,
        positions,
        False,
        execution_mode="spec_verify",
    )

    assert runner.model.forward_calls == 4
    assert graph.replays == 1


def test_spec_verify_capture_rolls_back_private_scratch_before_publication():
    runner = make_runner()
    cache_module = _install_spec_verify_runtime(runner)
    runner.graph_pool = None
    events = []
    live_slots = (15, 16, 31, 32)
    clone_pairs = []

    class Lease:
        block_ids = (100, 101, 102, 103)
        row_block_counts = (2, 2)

    class Pool:
        def acquire(
            self,
            *,
            active_batch_size,
            query_len,
            row_offsets,
        ):
            assert (active_batch_size, query_len) == (2, 2)
            assert row_offsets == (3, 3)
            events.append("scratch_acquire")
            return Lease()

        def rollback(self, lease):
            assert isinstance(lease, Lease)
            events.append("scratch_rollback")

    runner.spec_verify_capture_scratch_pool = Pool()
    runner._clone_spec_verify_capture_prefix_blocks = (
        lambda pairs: clone_pairs.extend(pairs)
    )

    class Model:
        def __init__(self):
            self.compute_logits_calls = 0
            self.capture_slots = None
            self.capture_block_tables = None

        def __call__(self, input_ids, positions, input_embeds=None):
            del input_ids, positions, input_embeds
            events.append("capture")
            capture_context = base.context.get_context()
            self.capture_slots = tuple(
                capture_context.slot_mapping.data
            )
            self.capture_block_tables = (
                capture_context.block_tables.data
            )
            return RuntimeTensor(
                (4, 8),
                dtype="torch.bfloat16",
                element_size=2,
                value="discarded-capture-output",
            )

        def compute_logits(self, outputs):
            del outputs
            self.compute_logits_calls += 1

    runner.model = Model()

    class Graph:
        pass

    class GraphContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    original_zeros = getattr(model_runner.torch, "zeros", None)
    original_tensor = getattr(model_runner.torch, "tensor", None)
    original_cuda = model_runner.torch.cuda

    def zeros(*shape, dtype=None, device=None):
        normalized = (
            (shape[0],)
            if len(shape) == 1
            else tuple(shape)
        )
        element_size = 8 if dtype == "torch.int64" else (
            2 if dtype == "torch.bfloat16" else 4
        )
        return RuntimeTensor(
            normalized,
            dtype=dtype,
            element_size=element_size,
            device=device,
        )

    def tensor(data, dtype=None, device=None):
        if data and isinstance(data[0], list):
            shape = (len(data), len(data[0]))
        else:
            shape = (len(data),)
        element_size = 8 if dtype == "torch.int64" else 4
        return RuntimeTensor(
            shape,
            dtype=dtype,
            element_size=element_size,
            device=device,
            data=data,
        )

    model_runner.torch.zeros = zeros
    model_runner.torch.tensor = tensor
    model_runner.torch.cuda = SimpleNamespace(
        get_device_capability=lambda device: (8, 0),
        CUDAGraph=Graph,
        graph=lambda graph, pool=None: GraphContext(),
        synchronize=lambda: None,
        memory_allocated=lambda: 100,
        memory_reserved=lambda: 200,
    )
    context = SimpleNamespace(
        mode="spec_verify",
        spec_verify_query_lens=(2, 2),
        slot_mapping=RuntimeTensor(
            (4,),
            dtype="torch.int32",
            element_size=4,
            data=list(live_slots),
        ),
        context_lens=RuntimeTensor(
            (2,),
            dtype="torch.int32",
            element_size=4,
            data=[5, 9],
        ),
        block_tables=RuntimeTensor(
            (2, 3),
            dtype="torch.int32",
            element_size=4,
            data=[[1, 2, 0], [3, 4, 5]],
        ),
        flash_attn_num_splits=16,
    )
    input_ids = RuntimeTensor(
        (4,),
        dtype="torch.int64",
        element_size=8,
        data=[10, 11, 12, 13],
    )
    positions = RuntimeTensor(
        (4,),
        dtype="torch.int64",
        element_size=8,
        data=[0, 1, 4, 5],
    )
    outputs = RuntimeTensor(
        (4, 8),
        dtype="torch.bfloat16",
        element_size=2,
    )
    identity = runner._build_spec_verify_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )
    runner.spec_verify_exact_cuda_graph_cache.observe_success(
        identity,
        estimated_static_bytes=(
            runner._estimate_spec_verify_graph_static_bytes(identity)
        ),
        step_id=1,
    )
    runner.spec_verify_exact_cuda_graph_cache.observe_success(
        identity,
        estimated_static_bytes=(
            runner._estimate_spec_verify_graph_static_bytes(identity)
        ),
        step_id=2,
    )
    original_commit = (
        runner.spec_verify_exact_cuda_graph_cache.commit_capture
    )

    def commit(entry):
        events.append("cache_publish")
        original_commit(entry)

    runner.spec_verify_exact_cuda_graph_cache.commit_capture = commit
    try:
        entry = runner._attempt_post_step_spec_verify_capture(
            identity=identity,
            live_input_ids=input_ids,
            live_positions=positions,
            live_context=context,
        )
    finally:
        if original_zeros is None:
            delattr(model_runner.torch, "zeros")
        else:
            model_runner.torch.zeros = original_zeros
        if original_tensor is None:
            delattr(model_runner.torch, "tensor")
        else:
            model_runner.torch.tensor = original_tensor
        model_runner.torch.cuda = original_cuda
        base.context.reset_context()

    assert entry is not None
    assert events == [
        "scratch_acquire",
        "capture",
        "scratch_rollback",
        "cache_publish",
    ]
    assert clone_pairs == [(1, 100), (4, 102)]
    assert runner.model.capture_slots == (403, 404, 411, 412)
    assert runner.model.capture_block_tables == [
        [100, 101, 0],
        [3, 102, 103],
    ]
    assert set(runner.model.capture_slots).isdisjoint(live_slots)
    assert runner.model.compute_logits_calls == 0
    assert entry.tensors["outputs"].value == (
        "discarded-capture-output"
    )


def test_spec_verify_capture_reports_partial_terminal_prefix_offsets():
    runner = make_runner()
    _install_spec_verify_runtime(runner)
    identity = SimpleNamespace(
        active_batch_size=2,
        query_len=2,
    )

    assert runner._spec_verify_capture_row_offsets(
        identity=identity,
        live_context=SimpleNamespace(
            context_lens=RuntimeTensor(
                (2,),
                dtype="torch.int32",
                element_size=4,
                data=[2, 6],
            )
        ),
    ) == (0, 0)

    assert runner._spec_verify_capture_row_offsets(
        identity=identity,
        live_context=SimpleNamespace(
            context_lens=RuntimeTensor(
                (2,),
                dtype="torch.int32",
                element_size=4,
                data=[3, 7],
            )
        ),
    ) == (1, 1)


def test_spec_verify_capture_clones_terminal_kv_and_auxiliary_state():
    runner = make_runner()
    copied = []

    class BlockView:
        def __init__(self, tensor_name, block_id):
            self.tensor_name = tensor_name
            self.block_id = block_id

        def copy_(self, source):
            copied.append(
                (
                    self.tensor_name,
                    source.block_id,
                    self.block_id,
                )
            )

    class BlockTensor:
        def __init__(self, tensor_name):
            self.tensor_name = tensor_name

        def __getitem__(self, index):
            assert index[:2] == (slice(None), slice(None))
            return BlockView(self.tensor_name, index[2])

    runner.kv_cache = BlockTensor("kv_cache")
    runner.kv_scale = BlockTensor("kv_scale")
    runner.kv_zero = BlockTensor("kv_zero")
    runner.kv_summary = BlockTensor("kv_summary")

    runner._clone_spec_verify_capture_prefix_blocks(
        ((1, 100), (4, 102))
    )

    assert copied == [
        ("kv_cache", 1, 100),
        ("kv_scale", 1, 100),
        ("kv_zero", 1, 100),
        ("kv_summary", 1, 100),
        ("kv_cache", 4, 102),
        ("kv_scale", 4, 102),
        ("kv_zero", 4, 102),
        ("kv_summary", 4, 102),
    ]


def test_spec_verify_capture_does_not_acquire_scratch_before_static_allocation():
    runner = make_runner()
    _install_spec_verify_runtime(runner)
    events = []

    class Lease:
        block_ids = (100,)
        row_block_counts = (1,)

    class Pool:
        def acquire(
            self,
            *,
            active_batch_size,
            query_len,
            row_offsets,
        ):
            assert (active_batch_size, query_len) == (1, 2)
            assert row_offsets == (0,)
            events.append("scratch_acquire")
            return Lease()

        def rollback(self, lease):
            assert isinstance(lease, Lease)
            events.append("scratch_rollback")

    runner.spec_verify_capture_scratch_pool = Pool()
    context = SimpleNamespace(
        mode="spec_verify",
        spec_verify_query_lens=(2,),
        slot_mapping=RuntimeTensor(
            (2,),
            dtype="torch.int32",
            element_size=4,
            data=[11, 12],
        ),
        context_lens=RuntimeTensor(
            (1,),
            dtype="torch.int32",
            element_size=4,
            data=[2],
        ),
        block_tables=RuntimeTensor(
            (1, 1),
            dtype="torch.int32",
            element_size=4,
            data=[[1]],
        ),
        flash_attn_num_splits=16,
    )
    input_ids = RuntimeTensor(
        (2,),
        dtype="torch.int64",
        element_size=8,
        data=[10, 11],
    )
    positions = RuntimeTensor(
        (2,),
        dtype="torch.int64",
        element_size=8,
        data=[0, 1],
    )
    outputs = RuntimeTensor(
        (2, 8),
        dtype="torch.bfloat16",
        element_size=2,
    )
    identity = runner._build_spec_verify_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )

    original_zeros = getattr(model_runner.torch, "zeros", None)
    original_cuda = model_runner.torch.cuda
    model_runner.torch.zeros = (
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("allocation failed"))
    )
    model_runner.torch.cuda = SimpleNamespace(
        CUDAGraph=lambda: object(),
    )
    try:
        with pytest.raises(RuntimeError, match="allocation failed"):
            runner._capture_spec_verify_graph(
                identity=identity,
                live_input_ids=input_ids,
                live_positions=positions,
                live_context=context,
            )
    finally:
        if original_zeros is None:
            delattr(model_runner.torch, "zeros")
        else:
            model_runner.torch.zeros = original_zeros
        model_runner.torch.cuda = original_cuda

    assert events == []


def test_spec_verify_replay_returns_transformer_outputs_without_logits():
    runner = make_runner()
    cache_module = _install_spec_verify_runtime(runner)
    context = SimpleNamespace(
        mode="spec_verify",
        spec_verify_query_lens=(3,),
        slot_mapping=RuntimeTensor(
            (3,),
            dtype="torch.int32",
            element_size=4,
            data=[11, 12, 13],
        ),
        context_lens=RuntimeTensor(
            (1,),
            dtype="torch.int32",
            element_size=4,
            data=[3],
        ),
        block_tables=RuntimeTensor(
            (1, 2),
            dtype="torch.int32",
            element_size=4,
            data=[[1, 2]],
        ),
        flash_attn_num_splits=16,
    )
    input_ids = _runtime_input()
    positions = _runtime_positions()
    outputs = RuntimeTensor(
        (3, 8),
        dtype="torch.bfloat16",
        element_size=2,
        value="graph-outputs",
    )
    identity = runner._build_spec_verify_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    class Model:
        def __init__(self):
            self.compute_logits_calls = 0

        def compute_logits(self, outputs):
            del outputs
            self.compute_logits_calls += 1

    graph = Graph()
    runner.model = Model()
    entry = cache_module.SpecVerifyExactCudaGraphEntry(
        identity=identity,
        identity_sha256=identity.sha256,
        graph=graph,
        tensors={
            "input_ids": RuntimeTensor(
                (3,),
                dtype="torch.int64",
                element_size=8,
            ),
            "positions": RuntimeTensor(
                (3,),
                dtype="torch.int64",
                element_size=8,
            ),
            "slot_mapping": RuntimeTensor(
                (3,),
                dtype="torch.int32",
                element_size=4,
            ),
            "context_lens": RuntimeTensor(
                (1,),
                dtype="torch.int32",
                element_size=4,
            ),
            "block_tables": RuntimeTensor(
                (1, 2),
                dtype="torch.int32",
                element_size=4,
            ),
            "outputs": outputs,
        },
        static_bytes=256,
        capture_duration_ns=100,
        allocated_delta_bytes=0,
        reserved_delta_bytes=0,
    )
    runner.spec_verify_exact_cuda_graph_cache.capturing[
        identity.sha256
    ] = entry.static_bytes
    runner.spec_verify_exact_cuda_graph_cache.commit_capture(entry)

    replayed = runner._replay_spec_verify_graph(
        entry,
        input_ids=input_ids,
        positions=positions,
        context=context,
    )

    assert replayed is outputs
    assert graph.replays == 1
    assert runner.model.compute_logits_calls == 0
    assert entry.replay_count == 1
    assert entry.in_flight_replays == 0
    assert base.context.get_context().mode == "decode"


@pytest.mark.parametrize(
    "failure_stage",
    ("replay", "synchronize"),
)
def test_spec_verify_replay_started_failure_has_no_eager_retry(
    failure_stage,
):
    runner = make_runner()
    cache_module = _install_spec_verify_runtime(runner)

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1
            if failure_stage == "replay":
                raise RuntimeError("replay failed")

    class Model:
        def __init__(self):
            self.eager_calls = 0
            self.compute_logits_calls = 0

        def __call__(self, *args, **kwargs):
            del args, kwargs
            self.eager_calls += 1
            return RuntimeTensor(
                (3, 8),
                dtype="torch.bfloat16",
                element_size=2,
            )

        def compute_logits(self, outputs):
            del outputs
            self.compute_logits_calls += 1

    graph = Graph()
    runner.model = Model()
    context, input_ids, positions, entry = (
        _install_ready_replay_entry(
            runner,
            cache_module,
            graph,
        )
    )

    def synchronize():
        if failure_stage == "synchronize":
            raise RuntimeError("synchronize failed")

    runner._synchronize_spec_verify_graph_replay = synchronize
    base.context.set_context(
        mode=context.mode,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        block_tables=context.block_tables,
        spec_verify_query_lens=context.spec_verify_query_lens,
        flash_attn_num_splits=context.flash_attn_num_splits,
    )
    try:
        with pytest.raises(
            cache_module.SpecVerifyGraphReplayError,
        ) as exc_info:
            runner.run_model(
                input_ids,
                positions,
                False,
                execution_mode="spec_verify",
            )
    finally:
        base.context.reset_context()

    assert exc_info.value.identity_sha256 == entry.identity_sha256
    assert isinstance(exc_info.value.cause, RuntimeError)
    assert graph.replays == 1
    assert runner.model.eager_calls == 0
    assert runner.model.compute_logits_calls == 0
    assert entry.state == "quarantined"
    assert entry.terminal_reason == "replay_failed"
    assert entry.in_flight_replays == 0
    assert entry.replay_count == 0

    base.context.set_context(
        mode=context.mode,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        block_tables=context.block_tables,
        spec_verify_query_lens=context.spec_verify_query_lens,
        flash_attn_num_splits=context.flash_attn_num_splits,
    )
    try:
        runner.run_model(
            input_ids,
            positions,
            False,
            execution_mode="spec_verify",
        )
    finally:
        base.context.reset_context()

    assert graph.replays == 1
    assert runner.model.eager_calls == 1
    assert runner.model.compute_logits_calls == 1


@pytest.mark.parametrize(
    (
        "failure_kind",
        "expected_reason",
        "expected_quarantine",
    ),
    (
        ("identity", "identity_drift", True),
        ("shape", "shape_drift", True),
        ("cache_state", "cache_state_drift", False),
        ("copy", "cache_state_drift", False),
    ),
)
def test_spec_verify_pre_replay_failure_falls_back_to_one_eager(
    failure_kind,
    expected_reason,
    expected_quarantine,
):
    runner = make_runner()
    cache_module = _install_spec_verify_runtime(runner)

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    class Model:
        def __init__(self):
            self.eager_calls = 0
            self.compute_logits_calls = 0

        def __call__(self, *args, **kwargs):
            del args, kwargs
            self.eager_calls += 1
            return RuntimeTensor(
                (3, 8),
                dtype="torch.bfloat16",
                element_size=2,
            )

        def compute_logits(self, outputs):
            del outputs
            self.compute_logits_calls += 1
            return "eager-logits"

    graph = Graph()
    runner.model = Model()
    context, input_ids, positions, entry = (
        _install_ready_replay_entry(
            runner,
            cache_module,
            graph,
        )
    )
    original_sha256 = entry.identity.sha256

    if failure_kind == "identity":
        entry.identity_sha256 = "0" * 64
    elif failure_kind == "shape":
        entry.tensors["positions"] = RuntimeTensor(
            (2,),
            dtype="torch.int64",
            element_size=8,
        )
    elif failure_kind == "cache_state":
        runner.spec_verify_exact_cuda_graph_cache.ready_entries.pop(
            original_sha256
        )
        runner._ready_spec_verify_graph_entry = (
            lambda **kwargs: entry
        )
    else:
        class CopyFailureTensor(RuntimeTensor):
            def copy_(self, value):
                del value
                raise RuntimeError("copy failed")

        entry.tensors["positions"] = CopyFailureTensor(
            (3,),
            dtype="torch.int64",
            element_size=8,
        )

    base.context.set_context(
        mode=context.mode,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        block_tables=context.block_tables,
        spec_verify_query_lens=context.spec_verify_query_lens,
        flash_attn_num_splits=context.flash_attn_num_splits,
    )
    try:
        result = runner.run_model(
            input_ids,
            positions,
            False,
            execution_mode="spec_verify",
        )
    finally:
        base.context.reset_context()

    assert result == "eager-logits"
    assert graph.replays == 0
    assert runner.model.eager_calls == 1
    assert runner.model.compute_logits_calls == 1
    event = runner.spec_verify_graph_dispatch_observation()
    assert event["dispatch"] == "eager"
    assert event["fallback_reason"] == expected_reason
    if expected_quarantine:
        assert entry.state == "quarantined"
        assert entry.terminal_reason == expected_reason
    else:
        assert entry.state == "ready"


def test_spec_verify_graph_helpers_do_not_finalize_live_transactions():
    source = "\n".join(
        (
            inspect.getsource(
                model_runner.ModelRunner._capture_spec_verify_graph
            ),
            inspect.getsource(
                model_runner.ModelRunner._replay_spec_verify_graph
            ),
        )
    )

    for forbidden in (
        "mark_speculative_kv_materialized",
        "mark_materialized",
        "rollback_speculative_kv_transaction",
        "precommit_speculative_kv_transaction",
        "commit_speculative_kv_transaction",
        "seal_speculative_kv_transaction",
    ):
        assert forbidden not in source
