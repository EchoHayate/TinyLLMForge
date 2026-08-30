"""Dependency-light tests for exact prefill CUDA Graph support."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import importlib.util
import ast
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"
GRAPH_PATH = (
    ROOT / "tinyvllm" / "engine" / "exact_prefill_cuda_graph.py"
)
MODEL_RUNNER_PATH = ROOT / "tinyvllm" / "engine" / "model_runner.py"


def load_real_config_class():
    module_name = "tinyvllm_exact_prefill_graph_config_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return types.SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        module = types.ModuleType(module_name)
        module.__file__ = os.fspath(CONFIG_PATH)
        sys.modules[module_name] = module
        source = CONFIG_PATH.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(CONFIG_PATH),
                "exec",
            ),
            module.__dict__,
        )
        return module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def load_exact_prefill_graph_module():
    assert GRAPH_PATH.exists(), (
        "exact prefill graph contract module has not been implemented"
    )
    module_name = "exact_prefill_cuda_graph_under_test"
    spec = importlib.util.spec_from_file_location(module_name, GRAPH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prefill_cuda_graph_config_defaults_and_normalization():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        default = Config(model=model)
        canonical = Config(
            model=model,
            prefill_cuda_graph_token_allowlist=[2048, 256, 2048],
        )

    assert default.prefill_cuda_graphs is False
    assert default.prefill_cuda_graph_token_allowlist == (256, 2048)
    assert canonical.prefill_cuda_graph_token_allowlist == (256, 2048)


@pytest.mark.parametrize(
    "overrides",
    (
        {"prefill_cuda_graphs": 1},
        {"prefill_cuda_graph_token_allowlist": ()},
        {"prefill_cuda_graph_token_allowlist": "256,2048"},
        {"prefill_cuda_graph_token_allowlist": (0, 256)},
        {"prefill_cuda_graph_token_allowlist": (True, 256)},
        {"prefill_cuda_graph_token_allowlist": (256.0,)},
    ),
)
def test_exact_prefill_cuda_graph_config_rejects_invalid_values(
    overrides,
):
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        with pytest.raises(ValueError):
            Config(model=model, **overrides)


def make_identity(module, *, token_count=256):
    return module.ExactPrefillGraphIdentity(
        token_count=token_count,
        active_batch_size=1,
        world_size=1,
        model_forward_kind="forward",
        attention_backend="flash_attn_varlen",
        attention_backend_version="2.6.3",
        input_dtype="torch.int64",
        hidden_dtype="torch.bfloat16",
        num_layers=28,
        hidden_size=1024,
        num_query_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_block_size=256,
        device_compute_capability=(8, 0),
    )


def eligible_kwargs():
    return {
        "enabled": True,
        "is_prefill": True,
        "tensor_parallel_size": 1,
        "world_size": 1,
        "sequence_count": 1,
        "input_token_count": 256,
        "query_len": 256,
        "key_len": 256,
        "has_prefix_block_table": False,
        "token_allowlist": (256, 2048),
        "input_embeddings_requested": False,
        "return_hidden_states": False,
        "cpu_offload": False,
        "kv_offload": False,
        "kv_quant_bits": 0,
        "compact_attention": False,
        "model_forward_kind": "forward",
    }


def test_exact_prefill_graph_identity_is_stable_and_shape_exact():
    module = load_exact_prefill_graph_module()
    first = make_identity(module)
    same = make_identity(module)
    wider = make_identity(module, token_count=2048)

    assert first.sha256 == same.sha256
    assert first.sha256 != wider.sha256
    assert len(first.sha256) == 64


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("enabled", False, "feature_disabled"),
        ("is_prefill", False, "not_prefill"),
        ("tensor_parallel_size", 2, "tensor_parallel_unsupported"),
        ("world_size", 2, "world_size_unsupported"),
        ("sequence_count", 2, "sequence_count_unsupported"),
        ("query_len", 255, "length_mismatch"),
        ("key_len", 255, "length_mismatch"),
        (
            "has_prefix_block_table",
            True,
            "prefix_block_table_present",
        ),
        (
            "input_token_count",
            512,
            "token_count_not_allowlisted",
        ),
        (
            "input_embeddings_requested",
            True,
            "input_embeddings_requested",
        ),
        (
            "return_hidden_states",
            True,
            "hidden_state_return_requested",
        ),
        ("cpu_offload", True, "cpu_offload_active"),
        ("kv_offload", True, "kv_offload_active"),
        ("kv_quant_bits", 8, "kv_quantization_active"),
        ("compact_attention", True, "compact_attention_active"),
        (
            "model_forward_kind",
            "run_step",
            "model_forward_unsupported",
        ),
    ),
)
def test_exact_prefill_eligibility_fails_closed_in_stable_order(
    field,
    value,
    reason,
):
    module = load_exact_prefill_graph_module()
    values = eligible_kwargs()
    values[field] = value
    if field == "input_token_count":
        values["query_len"] = value
        values["key_len"] = value
    decision = module.check_exact_prefill_graph_eligibility(**values)
    assert decision.eligible is False
    assert decision.fallback_reason == reason


def test_exact_prefill_eligibility_accepts_only_the_exact_shape():
    module = load_exact_prefill_graph_module()
    decision = module.check_exact_prefill_graph_eligibility(
        **eligible_kwargs()
    )
    assert decision.eligible is True
    assert decision.fallback_reason is None


def test_cache_accounts_capture_replay_and_per_shape_quarantine():
    module = load_exact_prefill_graph_module()
    cache = module.ExactPrefillCudaGraphCache(
        module.ExactPrefillCudaGraphCacheConfig(
            enabled=True,
            token_allowlist=(256, 2048),
        )
    )
    identity = make_identity(module)
    other = make_identity(module, token_count=2048)

    assert cache.begin_capture(identity) is True
    entry = module.ExactPrefillCudaGraphEntry(
        identity=identity,
        identity_sha256=identity.sha256,
        graph=object(),
        tensors={"input_ids": object()},
        static_bytes=4096,
        capture_duration_ns=12_000_000,
        allocated_delta_bytes=8_192,
        reserved_delta_bytes=16_384,
    )
    cache.commit_capture(entry)
    assert cache.ready_entry(identity) is entry
    cache.record_replay(entry, step=7)
    assert entry.replay_count == 1
    assert entry.last_replay_step == 7

    cache.quarantine(identity, "replay_failed")
    assert cache.ready_entry(identity) is None
    assert cache.begin_capture(identity) is False
    assert cache.begin_capture(other) is True

    summary = cache.summary()
    assert summary["capture_attempts"] == 2
    assert summary["capture_successes"] == 1
    assert summary["replays"] == 1
    assert summary["quarantines"] == 1
    assert summary["static_bytes"] == 4096
    assert summary["allocated_delta_bytes"] == 8192
    assert summary["reserved_delta_bytes"] == 16384
    assert summary["total_capture_ns"] == 12_000_000
    assert summary["quarantined"][identity.sha256] == "replay_failed"


def test_cache_records_preidentity_capture_errors_per_token_shape():
    module = load_exact_prefill_graph_module()
    cache = module.ExactPrefillCudaGraphCache(
        module.ExactPrefillCudaGraphCacheConfig(
            enabled=True,
            token_allowlist=(256, 2048),
        )
    )

    cache.record_capture_error(
        256,
        "RuntimeError: synthetic capture failure",
    )
    cache.record_capture_error(
        256,
        "ValueError: later failure must not replace first",
    )

    summary = cache.summary()
    assert summary["capture_failures"] == 1
    assert summary["capture_errors_by_token"] == {
        "256": "RuntimeError: synthetic capture failure",
    }


def _model_runner_class_node():
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=os.fspath(MODEL_RUNNER_PATH),
    )
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )


def _model_runner_method_node(name):
    return next(
        node
        for node in _model_runner_class_node().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_model_runner_initializes_and_captures_prefill_graphs_after_kv():
    source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    assert (
        "from tinyvllm.engine.exact_prefill_cuda_graph import ("
        in source
    )
    init_source = ast.get_source_segment(
        source,
        _model_runner_method_node("__init__"),
    )
    assert init_source is not None
    cache_init = init_source.index(
        "self.exact_prefill_cuda_graph_cache ="
    )
    kv_allocation = init_source.index("self.allocate_kv_cache()")
    capture = init_source.index(
        "self.capture_exact_prefill_cudagraphs()"
    )
    assert cache_init < kv_allocation < capture


def test_model_runner_declares_exact_prefill_capture_and_replay_methods():
    expected = {
        "_build_exact_prefill_graph_identity",
        "_capture_exact_prefill_graph",
        "capture_exact_prefill_cudagraphs",
        "_replay_exact_prefill_graph",
        "_try_replay_exact_prefill_graph",
    }
    actual = {
        node.name
        for node in _model_runner_class_node().body
        if isinstance(node, ast.FunctionDef)
    }
    assert expected <= actual


def test_startup_prefill_capture_runs_under_inference_mode():
    method = _model_runner_method_node(
        "capture_exact_prefill_cudagraphs"
    )
    decorators = {
        ast.unparse(decorator)
        for decorator in method.decorator_list
    }
    assert "torch.inference_mode()" in decorators


def test_run_model_attempts_prefill_replay_before_generic_eager_fallback():
    source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    run_source = ast.get_source_segment(
        source,
        _model_runner_method_node("run_model"),
    )
    assert run_source is not None
    replay = run_source.index(
        "self._try_replay_exact_prefill_graph("
    )
    generic_eager = run_source.index(
        "if (is_prefill or spec_verify_active"
    )
    assert replay < generic_eager


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype,
        device="cuda:0",
        element_size=4,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._element_size = element_size
        self.copied_from = None

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self._element_size

    def copy_(self, value):
        self.copied_from = value
        return self


class FakeGraph:
    def __init__(self, error=None):
        self.error = error
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1
        if self.error is not None:
            raise self.error


def make_model_runner_fixture(*, graph_error=None):
    from tools import test_model_runner_spec_verify as base

    module = base.model_runner
    module.torch.cuda.get_device_capability = lambda _device: (8, 0)
    runner = object.__new__(module.ModelRunner)
    runner.config = types.SimpleNamespace(
        prefill_cuda_graphs=True,
        prefill_cuda_graph_token_allowlist=(256, 2048),
        tensor_parallel_size=1,
        cpu_offload=False,
        kv_offload_mvp0=False,
        kv_quant_bits=0,
        am_compact_blocks=0,
        quest_top_k_blocks=-1,
        kv_cartridge_blocks=0,
        max_num_prefill_tokens_per_step=0,
        hf_config=types.SimpleNamespace(
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        ),
    )
    runner.world_size = 1
    runner.block_size = 256
    runner.model = types.SimpleNamespace()
    runner.kv_cache = types.SimpleNamespace(device="cuda:0")
    runner._prefill_cuda_graph_step_id = 0
    cache = module.ExactPrefillCudaGraphCache(
        module.ExactPrefillCudaGraphCacheConfig(
            enabled=True,
            token_allowlist=(256, 2048),
        )
    )
    runner.exact_prefill_cuda_graph_cache = cache
    context = types.SimpleNamespace(
        mode="prefill",
        max_seqlen_q=256,
        max_seqlen_k=256,
        block_tables=None,
        slot_mapping=FakeTensor((256,), dtype="torch.int32"),
        cu_seqlens_q=FakeTensor((2,), dtype="torch.int32"),
        cu_seqlens_k=FakeTensor((2,), dtype="torch.int32"),
    )
    input_ids = FakeTensor(
        (256,),
        dtype="torch.int64",
        element_size=8,
    )
    positions = FakeTensor(
        (256,),
        dtype="torch.int64",
        element_size=8,
    )
    outputs = FakeTensor(
        (256, 1024),
        dtype="torch.bfloat16",
        element_size=2,
    )
    identity = runner._build_exact_prefill_graph_identity(
        input_ids=input_ids,
        outputs=outputs,
        context=context,
    )
    graph = FakeGraph(graph_error)
    tensors = {
        "input_ids": FakeTensor(
            (256,),
            dtype="torch.int64",
            element_size=8,
        ),
        "positions": FakeTensor(
            (256,),
            dtype="torch.int64",
            element_size=8,
        ),
        "slot_mapping": FakeTensor(
            (256,),
            dtype="torch.int32",
        ),
        "cu_seqlens_q": FakeTensor((2,), dtype="torch.int32"),
        "cu_seqlens_k": FakeTensor((2,), dtype="torch.int32"),
        "outputs": outputs,
    }
    entry = module.ExactPrefillCudaGraphEntry(
        identity=identity,
        identity_sha256=identity.sha256,
        graph=graph,
        tensors=tensors,
        static_bytes=1,
        capture_duration_ns=1,
        allocated_delta_bytes=1,
        reserved_delta_bytes=1,
    )
    assert cache.begin_capture(identity) is True
    cache.commit_capture(entry)
    return runner, context, input_ids, positions, entry


def test_model_runner_replays_and_copies_all_live_prefill_metadata():
    runner, context, input_ids, positions, entry = (
        make_model_runner_fixture()
    )
    result = runner._try_replay_exact_prefill_graph(
        input_ids=input_ids,
        positions=positions,
        is_prefill=True,
        input_embeds=None,
        return_hidden=False,
        context=context,
    )

    assert result is entry.tensors["outputs"]
    assert entry.graph.replay_count == 1
    assert entry.replay_count == 1
    assert entry.tensors["input_ids"].copied_from is input_ids
    assert entry.tensors["positions"].copied_from is positions
    assert (
        entry.tensors["slot_mapping"].copied_from
        is context.slot_mapping
    )
    assert (
        entry.tensors["cu_seqlens_q"].copied_from
        is context.cu_seqlens_q
    )
    assert (
        entry.tensors["cu_seqlens_k"].copied_from
        is context.cu_seqlens_k
    )


def test_model_runner_replay_failure_quarantines_shape_without_eager_retry():
    from tools import test_model_runner_spec_verify as base

    runner, context, input_ids, positions, entry = (
        make_model_runner_fixture(
            graph_error=RuntimeError("synthetic replay failure")
        )
    )
    with pytest.raises(
        base.model_runner.ExactPrefillGraphReplayError,
        match="synthetic replay failure",
    ):
        runner._try_replay_exact_prefill_graph(
            input_ids=input_ids,
            positions=positions,
            is_prefill=True,
            input_embeds=None,
            return_hidden=False,
            context=context,
        )
    summary = runner.exact_prefill_cuda_graph_cache.summary()
    assert summary["replay_failures"] == 1
    assert summary["quarantined"][entry.identity_sha256] == (
        "replay_failed"
    )
