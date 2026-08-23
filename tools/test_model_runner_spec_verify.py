"""Dependency-light tests for ModelRunner native verifier preparation."""

from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from types import SimpleNamespace

try:
    import pytest
except ModuleNotFoundError:
    if __name__ != "__main__":
        raise

    class _Raises:
        def __init__(self, expected, *, match=None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, _traceback):
            if exception_type is None:
                raise AssertionError(
                    f"did not raise {self.expected!r}"
                )
            if not issubclass(exception_type, self.expected):
                return False
            if (
                self.match is not None
                and re.search(self.match, str(exception)) is None
            ):
                raise AssertionError(
                    f"{exception!r} does not match {self.match!r}"
                )
            return True

    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda function: function

    class _PytestCompat:
        mark = _Mark()

        @staticmethod
        def raises(expected, *, match=None):
            return _Raises(expected, match=match)

        @staticmethod
        def skip(reason):
            raise RuntimeError(f"skipped: {reason}")

    pytest = _PytestCompat()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_VERIFIER_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "speculative",
    "verifier.py",
)
_CONTEXT_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "utils",
    "context.py",
)
_MODEL_RUNNER_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "model_runner.py",
)
_CONFIG_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "config.py",
)
_GREEDY_SAMPLING_FAST_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "greedy_sampling_fast_path.py",
)
_GRAPH_RESIDENT_GREEDY_TAIL_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "graph_resident_greedy_tail.py",
)
_EXACT_GREEDY_DECODE_BURST_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "exact_greedy_decode_burst.py",
)
_SPLIT_POLICY_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "flash_attn_split_policy.py",
)
_EXACT_CACHE_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "exact_cuda_graph_cache.py",
)
_SPEC_VERIFY_EXACT_CACHE_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "spec_verify_exact_cuda_graph_cache.py",
)
_COMMAND_ACK_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "model_runner_command_ack.py",
)
_PREEXISTING_COMMAND_ACK_MODULE = sys.modules.get(
    "tinyvllm.engine.model_runner_command_ack"
)
_SPEC_VERIFY_TRACE_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "spec_verify_trace.py",
)


class FakeTensor:
    def __init__(self, values, *, device="cuda:0", element_size=2):
        self.values = values
        self.device = device
        self._element_size = element_size

    def size(self, dim=None):
        if dim is None:
            if isinstance(self.values, list):
                return (len(self.values),)
            return ()
        if dim == 0:
            return len(self.values)
        if (
            dim == 1
            and isinstance(self.values, list)
            and self.values
            and isinstance(self.values[0], list)
        ):
            return len(self.values[0])
        raise IndexError(dim)

    def element_size(self):
        return self._element_size


class FakeIndexedTensor:
    def __init__(self, values, trace=None):
        self.values = values
        self.trace = [] if trace is None else trace

    def __getitem__(self, index):
        self.trace.append(("getitem", index))
        return FakeIndexedTensor(("selected", index), self.trace)

    def detach(self):
        self.trace.append(("detach", None))
        return self

    def float(self):
        self.trace.append(("float", None))
        return self

    def cpu(self):
        self.trace.append(("cpu", None))
        return self

    def clone(self):
        self.trace.append(("clone", None))
        return FakeIndexedTensor(("cloned", self.values), self.trace)


class FakeGraphBuffer:
    def __init__(self, values=None):
        self.values = values
        self.zero_calls = 0
        self.assignments = []

    def zero_(self):
        self.zero_calls += 1

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def __setitem__(self, index, value):
        self.assignments.append(
            (index, getattr(value, "values", value))
        )

    def copy_(self, value):
        self.values = getattr(value, "values", value)
        return self


class FakeCaptureTensor(FakeGraphBuffer):
    def __init__(self, shape, *, element_size=4):
        super().__init__(None)
        self.shape = (
            (shape,)
            if isinstance(shape, int)
            else tuple(shape)
        )
        self._element_size = element_size

    def __getitem__(self, index):
        return self

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return self._element_size


def _install_module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_source_module(name: str, path: str):
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    source = open(path).read()
    code = compile(
        source,
        path,
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, module.__dict__)
    return module


def _load_model_runner_module():
    original_torch = sys.modules.get("torch")
    original_torch_distributed = sys.modules.get(
        "torch.distributed"
    )

    @dataclass(frozen=True)
    class HybridStateLease:
        slot_id: int
        generation: int
        request_id: int

    torch_module = _install_module(
        "torch",
        Tensor=FakeTensor,
        int64="int64",
        int32="int32",
        float32="float32",
        long="long",
    )
    torch_module.tensor = (
        lambda values, device=None, dtype=None: FakeIndexedTensor(
            {
                "values": list(values),
                "device": device,
                "dtype": dtype,
            }
        )
    )
    torch_module.inference_mode = lambda: (lambda function: function)
    torch_module.cuda = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(
            multi_processor_count=108,
        ),
    )
    distributed_module = _install_module("torch.distributed")
    torch_module.distributed = distributed_module

    tinyvllm_package = _install_module("tinyvllm")
    tinyvllm_package.__path__ = [
        os.path.join(_REPO_ROOT, "tinyvllm")
    ]
    for package_name, package_path in (
        ("tinyvllm.speculative", "speculative"),
        ("tinyvllm.engine", "engine"),
        ("tinyvllm.models", "models"),
        ("tinyvllm.utils", "utils"),
        ("tinyvllm.layers", "layers"),
    ):
        package = _install_module(package_name)
        package.__path__ = [
            os.path.join(
                _REPO_ROOT,
                "tinyvllm",
                package_path,
            )
        ]

    verifier_spec = importlib.util.spec_from_file_location(
        "tinyvllm.speculative.verifier",
        _VERIFIER_PATH,
    )
    verifier_module = importlib.util.module_from_spec(verifier_spec)
    sys.modules["tinyvllm.speculative.verifier"] = verifier_module
    verifier_spec.loader.exec_module(verifier_module)

    context_module = _load_source_module(
        "tinyvllm.utils.context",
        _CONTEXT_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.flash_attn_split_policy",
        _SPLIT_POLICY_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.exact_cuda_graph_cache",
        _EXACT_CACHE_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.spec_verify_exact_cuda_graph_cache",
        _SPEC_VERIFY_EXACT_CACHE_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.greedy_sampling_fast_path",
        _GREEDY_SAMPLING_FAST_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.graph_resident_greedy_tail",
        _GRAPH_RESIDENT_GREEDY_TAIL_PATH,
    )
    _load_source_module(
        "tinyvllm.engine.exact_greedy_decode_burst",
        _EXACT_GREEDY_DECODE_BURST_PATH,
    )
    if "tinyvllm.engine.model_runner_command_ack" not in sys.modules:
        _load_source_module(
            "tinyvllm.engine.model_runner_command_ack",
            _COMMAND_ACK_PATH,
        )
    _load_source_module(
        "tinyvllm.engine.spec_verify_trace",
        _SPEC_VERIFY_TRACE_PATH,
    )
    _install_module(
        "tinyvllm.engine.decode_internal_profiler",
        DecodeInternalProfiler=type(
            "DecodeInternalProfiler",
            (),
            {
                "disabled": classmethod(
                    lambda cls, rank: SimpleNamespace(
                        rank=rank,
                        finalize=lambda: {},
                    )
                ),
            },
        ),
        run_profiled_step=(
            lambda _profiler, **kwargs: kwargs["call"]()
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
        Qwen35HybridPrefixRestoreParticipant=type(
            "Qwen35HybridPrefixRestoreParticipant",
            (),
            {},
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
        Qwen35HybridPrefixPublicationParticipant=type(
            "Qwen35HybridPrefixPublicationParticipant",
            (),
            {},
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_owner",
        Qwen35HybridPrefixRestoreOwner=type(
            "Qwen35HybridPrefixRestoreOwner",
            (),
            {},
        ),
        build_qwen35_hybrid_prefix_restore_owner=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_model_owner",
        Qwen35HybridModelOwner=type(
            "Qwen35HybridModelOwner",
            (),
            {},
        ),
        build_qwen35_hybrid_model_owner=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_model_publication",
        Qwen35HybridModelOwnerPublicationSlot=type(
            "Qwen35HybridModelOwnerPublicationSlot",
            (),
            {},
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity",
        bind_qwen35_hybrid_prefix_runtime_identity=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_recurrent_capture",
        Qwen35RecurrentCaptureSession=type(
            "Qwen35RecurrentCaptureSession",
            (),
            {},
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_recurrent_capture_contract",
        CAPTURE_IDENTITY_SCHEMA_VERSION=1,
        validate_run_identity=lambda *args, **kwargs: None,
    )
    _install_module(
        "tinyvllm.engine.qwen35_speculative_state",
        Qwen35SpeculativeStateOwner=type(
            "Qwen35SpeculativeStateOwner",
            (),
            {
                "__init__": lambda self, transaction: setattr(
                    self,
                    "state_transaction",
                    transaction,
                ),
                "active": False,
            },
        ),
    )
    _install_module(
        "tinyvllm.models.qwen35_checkpoint_streaming",
        Qwen35LoadedCheckpointCandidate=type(
            "Qwen35LoadedCheckpointCandidate",
            (),
            {},
        ),
        load_qwen35_fresh_checkpoint_candidate=(
            lambda *args, **kwargs: None
        ),
        move_qwen35_loaded_checkpoint_candidate_to_device=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module(
        "tinyvllm.models.qwen35_checkpoint_metadata",
        Qwen35CheckpointShardIdentity=type(
            "Qwen35CheckpointShardIdentity",
            (),
            {},
        ),
        read_qwen35_checkpoint_metadata=lambda *args, **kwargs: None,
    )
    _install_module(
        "tinyvllm.models.qwen35_checkpoint",
        build_qwen35_checkpoint_tensor_plan=lambda *args, **kwargs: None,
    )
    _install_module(
        "tinyvllm.models.qwen35_checkpoint_candidate_factory",
        prepare_qwen35_checkpoint_candidate_target=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module(
        "tinyvllm.engine.qwen35_hybrid_state",
        build_qwen35_hybrid_state_layout=lambda *args, **kwargs: None,
    )
    _install_module(
        "tinyvllm.layers.attention",
        Attention=type("Attention", (), {}),
    )
    _install_module(
        "tinyvllm.models.qwen35_checkpoint_worker",
        validate_qwen35_checkpoint_candidate_load_request=(
            lambda *args, **kwargs: None
        ),
    )
    _install_module("flash_attn", __version__="2.6.3")

    _install_module("tinyvllm.config", Config=type("Config", (), {}))
    _install_module(
        "tinyvllm.engine.sequence",
        Sequence=type("Sequence", (), {}),
    )
    _install_module(
        "tinyvllm.engine.hybrid_state",
        HybridStateLease=HybridStateLease,
        HybridStateTensorPool=type("HybridStateTensorPool", (), {}),
    )
    _install_module(
        "tinyvllm.models.qwen3",
        Qwen3ForCausalLM=type("Qwen3ForCausalLM", (), {}),
    )
    _install_module("tinyvllm.utils.loader", load_model=lambda *args: None)
    _install_module(
        "tinyvllm.utils.cpu_offload",
        apply_cpu_offload=lambda *args: None,
    )
    _install_module(
        "tinyvllm.layers.linear",
        configure_linear_execution_rows=lambda *args: None,
        set_quant_config=lambda *args: None,
    )
    _install_module(
        "tinyvllm.layers.sampler",
        Sampler=type("Sampler", (), {}),
    )
    _install_module(
        "tinyvllm.engine.kv_cartridge",
        compress_decode_block_table_rows=lambda rows, lengths, *args: (
            rows,
            lengths,
        ),
        should_use_kv_cartridge=lambda *args: False,
    )

    model_runner = _load_source_module(
        "model_runner_spec_verify_under_test",
        _MODEL_RUNNER_PATH,
    )
    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch
    if original_torch_distributed is None:
        sys.modules.pop("torch.distributed", None)
    else:
        sys.modules[
            "torch.distributed"
        ] = original_torch_distributed
    return model_runner, context_module


model_runner, context = _load_model_runner_module()
ModelRunner = model_runner.ModelRunner
graph_tail_module = sys.modules[
    "tinyvllm.engine.graph_resident_greedy_tail"
]
GraphResidentGreedyTailReplay = (
    graph_tail_module.GraphResidentGreedyTailReplay
)
verifier = sys.modules["tinyvllm.speculative.verifier"]
SpecVerifyTraceRecorder = sys.modules[
    "tinyvllm.engine.spec_verify_trace"
].SpecVerifyTraceRecorder
from tinyvllm.engine.speculative_residency import (  # noqa: E402
    KVBlockIdentityRow,
    SpeculativeResidencyPrecommitRow,
    SpeculativeResidencyPrepareRow,
    SpeculativeResidencyResult,
)
from tinyvllm.engine.speculative_proposal_executor import (  # noqa: E402
    ModelRunnerProposalExecutorRegistry,
)
from tinyvllm.speculative.adapter import (  # noqa: E402
    DraftCapabilities,
    DraftProposal,
)
from tinyvllm.speculative.batch_runtime import (  # noqa: E402
    FirstTargetProposalResult,
)


def test_dependency_light_loader_preserves_preexisting_command_ack_module():
    if _PREEXISTING_COMMAND_ACK_MODULE is None:
        pytest.skip("no preexisting command-ack module")
    assert (
        sys.modules["tinyvllm.engine.model_runner_command_ack"]
        is _PREEXISTING_COMMAND_ACK_MODULE
    )


def test_init_prepares_cuda_graph_dispatch_state_before_warmup():
    tree = ast.parse(open(_MODEL_RUNNER_PATH).read())
    model_runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init_function = next(
        node
        for node in model_runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    statement_index_by_attribute = {}
    warmup_index = None
    for statement_index, statement in enumerate(init_function.body):
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                statement_index_by_attribute.setdefault(
                    node.attr,
                    statement_index,
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "warmup_model"
            ):
                warmup_index = statement_index

    assert warmup_index is not None
    for attribute in (
        "exact_cuda_graph_cache",
        "last_cuda_graph_dispatch_event",
        "_cuda_graph_step_id",
        "_cuda_graph_request_ids_hash",
    ):
        assert statement_index_by_attribute[attribute] < warmup_index


def test_prefill_window_reserves_current_write_blocks_without_mutating_decode_window():
    requested_decode_window = 2

    effective_prefill_window = (
        model_runner._resolve_blockwise_prefill_window_blocks(
            requested_decode_window,
            gpu_blocks=2,
            write_blocks=[7],
        )
    )

    assert effective_prefill_window == 1
    assert requested_decode_window == 2


def make_runner(**overrides):
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    runner.rank = 0
    runner.world_size = 1
    runner.kv_offload = None
    runner.hybrid_state_runtime_bridge = None
    runner.qwen35_speculative_state_owner = None
    runner._speculative_side_state_handle = None
    runner._speculative_side_state_leases_by_sequence = {}
    runner.enforce_eager = False
    config = {
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "kv_offload_blockwise_blocks": 0,
        "quest_top_k_blocks": -1,
        "quest_min_seq_len": 0,
        "am_compact_blocks": 0,
        "am_compact_min_seq_len": 0,
        "am_compact_selector": "highest",
        "am_compact_score_method": "rms",
        "am_compact_beta_bound": 3.0,
        "am_compact_ridge_lambda": 1e-6,
        "am_omp_candidate_pool_size": 0,
        "am_compact_cache_refresh_interval": 0,
        "am_compact_num_clusters": 1,
        "am_compact_route_top_k": 1,
        "am_compact_num_key_spans": 1,
        "am_compact_decode_refit": False,
        "am_compact_decode_refit_mode": "full",
        "am_compact_decode_refit_interval": 1,
        "am_compact_skip_first_layers": 0,
        "am_compact_skip_last_layers": 0,
        "am_compact_enable_layers": None,
        "am_compact_layer_stride": 1,
        "kv_cartridge_blocks": 0,
        "kv_cartridge_min_seq_len": 0,
        "chunked_prefill_mixed_batch": False,
        "cpu_offload": False,
        "replay_aware_decode_metadata": False,
        "zero_temperature_greedy_fast_path": False,
        "graph_resident_greedy_tail": False,
        "exact_greedy_decode_burst": False,
        "exact_greedy_decode_burst_continuation": False,
        "exact_greedy_decode_burst_split_phase": False,
        "exact_greedy_decode_burst_tokens": 4,
        "multi_sequence_cuda_graphs": False,
        "multi_sequence_cuda_graph_batch_allowlist": (2, 4, 8),
        "spec_verify_cuda_graphs": False,
    }
    config.update(overrides)
    runner.config = SimpleNamespace(**config)
    cache_module = sys.modules[
        "tinyvllm.engine.exact_cuda_graph_cache"
    ]
    runner.exact_cuda_graph_cache = cache_module.ExactCudaGraphCache(
        cache_module.ExactCudaGraphCacheConfig(
            enabled=runner.config.multi_sequence_cuda_graphs,
            batch_allowlist=(2, 4, 8),
            min_observations=3,
            max_entries=8,
            max_static_bytes=64 * 1024 * 1024,
            max_reserved_bytes=512 * 1024 * 1024,
            max_total_capture_ns=5_000_000_000,
            max_single_capture_ns=2_000_000_000,
        )
    )
    runner.last_cuda_graph_dispatch_event = None
    runner._cuda_graph_step_id = 0
    runner._cuda_graph_request_ids_hash = "request-hash"
    runner.decode_internal_profiler = SimpleNamespace()
    greedy_module = sys.modules[
        "tinyvllm.engine.greedy_sampling_fast_path"
    ]
    runner.greedy_sampling_fast_path_stats = (
        greedy_module.GreedySamplingFastPathStats()
    )
    graph_tail_module = sys.modules[
        "tinyvllm.engine.graph_resident_greedy_tail"
    ]
    runner.graph_resident_greedy_tail = None
    runner.graph_resident_greedy_tail_stats = (
        graph_tail_module.GraphResidentGreedyTailStats()
    )
    burst_module = sys.modules[
        "tinyvllm.engine.exact_greedy_decode_burst"
    ]
    runner.exact_greedy_decode_burst_graph = None
    runner.exact_greedy_decode_burst_correctness_graph = None
    runner.exact_greedy_decode_burst_split_phase_backend = None
    runner.exact_greedy_decode_burst_split_phase_correctness_backend = (
        None
    )
    runner.exact_greedy_decode_burst_stats = (
        burst_module.ExactGreedyDecodeBurstStats()
    )
    runner._ordinary_graph_generation = 0
    runner.qwen35_recurrent_capture_session = None
    runner._spec_verify_trace = SpecVerifyTraceRecorder(
        rank=runner.rank,
        block_size=runner.block_size,
    )
    runner._list_to_cuda = (
        lambda data, name, dtype: FakeTensor(list(data))
    )
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables": FakeTensor(
            [list(row) for row in rows]
        )
    )
    return runner


def _trace_ready_runner(rank=0):
    runner = make_runner()
    runner.rank = rank
    runner.block_size = 256
    runner._spec_verify_trace = SpecVerifyTraceRecorder(
        rank=rank,
        block_size=256,
    )
    return runner


def test_spec_verify_trace_is_default_off():
    runner = _trace_ready_runner()
    assert runner.drain_spec_verify_trace_rows() == ()


def test_spec_verify_trace_lifecycle_is_explicit():
    runner = _trace_ready_runner()

    assert runner.enable_spec_verify_trace_recording(True) == {
        "rank": 0,
        "enabled": True,
    }
    assert runner.set_spec_verify_trace_context(
        "native_mtp",
        1,
        4,
    ) == {
        "rank": 0,
        "policy": "native_mtp",
        "batch_size": 1,
        "engine_step": 4,
    }
    assert runner.enable_spec_verify_trace_recording(False) == {
        "rank": 0,
        "enabled": False,
    }


def test_trace_block_identities_use_list_backed_generations():
    runner = _trace_ready_runner()
    runner.enable_spec_verify_trace_recording(True)
    runner.kv_offload = SimpleNamespace(
        bound_generations=[None, None, None, 7, 8, None],
    )

    assert runner._trace_block_identities((3, 4)) == (
        (3, 7),
        (4, 8),
    )
    with pytest.raises(
        RuntimeError,
        match="trace block generation is missing",
    ):
        runner._trace_block_identities((3, 5))


def test_bind_kv_block_identity_rows_matches_sequence_tables():
    calls = []
    runner = make_runner(kv_offload_mvp0=True)
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=(
            lambda block_id, generation:
            calls.append((block_id, generation))
        )
    )
    sequence = SimpleNamespace(
        seq_id=7,
        block_table=[1, 3],
    )

    runner.bind_kv_block_identity_rows(
        (sequence,),
        (
            KVBlockIdentityRow(
                sequence_id=7,
                block_identities=((1, 4), (3, 2)),
            ),
        ),
    )

    assert calls == [(1, 4), (3, 2)]


def test_bind_kv_block_identity_rows_rejects_table_mismatch():
    runner = make_runner(kv_offload_mvp0=True)
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=lambda *args: None
    )
    sequence = SimpleNamespace(
        seq_id=7,
        block_table=[1, 3],
    )

    with pytest.raises(ValueError, match="table mismatch"):
        runner.bind_kv_block_identity_rows(
            (sequence,),
            (
                KVBlockIdentityRow(
                    sequence_id=7,
                    block_identities=((1, 4), (2, 2)),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("method_name", "participant_method", "args", "operation", "status"),
    (
        (
            "prepare_speculative_residency_batch",
            "prepare_batch",
            (
                41,
                (
                    SpeculativeResidencyPrepareRow(
                        sequence_id=7,
                        original_block_identities=((1, 1),),
                        reserved_block_identities=((2, 1),),
                        proxy_block_table=(1, 2),
                        logical_slots=(3, 4),
                    ),
                ),
            ),
            "prepare",
            "prepared",
        ),
        (
            "precommit_speculative_residency_batch",
            "precommit_batch",
            (
                41,
                (
                    SpeculativeResidencyPrecommitRow(
                        sequence_id=7,
                        committed_block_identities=((2, 1),),
                        rejected_block_identities=(),
                        accepted_materialized_end=5,
                    ),
                ),
            ),
            "precommit",
            "precommitted",
        ),
        (
            "rollback_speculative_residency_batch",
            "rollback_batch",
            (41,),
            "rollback",
            "rolled_back",
        ),
        (
            "seal_speculative_residency_batch",
            "seal_batch",
            (41,),
            "seal",
            "sealed",
        ),
    ),
)
def test_speculative_residency_rpcs_return_exact_result_dict(
    method_name,
    participant_method,
    args,
    operation,
    status,
):
    calls = []
    result = SpeculativeResidencyResult(
        ticket_id=41,
        participant_id=0,
        operation=operation,
        status=status,
        sequence_ids=(7,),
        committed_block_identities=((2, 1),),
        rejected_block_identities=((3, 1),),
        detail="",
    )
    participant = SimpleNamespace()
    setattr(
        participant,
        participant_method,
        lambda *received, **received_kwargs:
        calls.append((received, received_kwargs)) or result,
    )
    runner = make_runner()
    runner.speculative_residency = participant

    payload = getattr(runner, method_name)(*args)

    expected_kwargs = (
        {"stage_all_original_blocks": True}
        if method_name == "prepare_speculative_residency_batch"
        else {}
    )
    assert calls == [(args, expected_kwargs)]
    assert payload == {
        "ticket_id": 41,
        "participant_id": 0,
        "operation": operation,
        "status": status,
        "sequence_ids": (7,),
        "committed_block_identities": ((2, 1),),
        "rejected_block_identities": ((3, 1),),
        "detail": "",
    }


def test_speculative_residency_rpc_requires_offload_participant():
    runner = make_runner()
    runner.speculative_residency = None

    with pytest.raises(RuntimeError, match="kv_offload_mvp0"):
        runner.rollback_speculative_residency_batch(41)


@pytest.mark.parametrize(
    (
        "blockwise_decode",
        "expected_stage_all_original_blocks",
    ),
    (
        (False, True),
        (True, False),
    ),
)
def test_speculative_residency_prepare_selects_history_staging_policy(
    blockwise_decode,
    expected_stage_all_original_blocks,
):
    calls = []
    result = SpeculativeResidencyResult(
        ticket_id=41,
        participant_id=0,
        operation="prepare",
        status="prepared",
        sequence_ids=(7,),
    )

    def prepare_batch(ticket_id, rows, **kwargs):
        calls.append((ticket_id, rows, kwargs))
        return result

    runner = make_runner(
        kv_offload_mvp0=True,
        kv_offload_blockwise_decode=blockwise_decode,
    )
    runner.speculative_residency = SimpleNamespace(
        prepare_batch=prepare_batch,
    )
    rows = (
        SpeculativeResidencyPrepareRow(
            sequence_id=7,
            original_block_identities=((1, 1),),
            reserved_block_identities=((2, 1),),
            proxy_block_table=(1, 2),
            logical_slots=(3, 4),
        ),
    )

    payload = runner.prepare_speculative_residency_batch(
        41,
        rows,
    )

    assert calls == [
        (
            41,
            rows,
            {
                "stage_all_original_blocks": (
                    expected_stage_all_original_blocks
                )
            },
        )
    ]
    assert payload["status"] == "prepared"


def make_sequence(seq_id):
    return SimpleNamespace(
        seq_id=seq_id,
        hybrid_state_slot_id=-1,
        hybrid_state_generation=0,
    )


class PrefillSequence(list):
    def __init__(
        self,
        token_ids,
        *,
        chunk_start,
        chunk_end,
        block_table,
    ):
        super().__init__(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = chunk_start
        self.prefill_chunk_start = chunk_start
        self.prefill_chunk_end = chunk_end
        self.block_table = block_table


class DecodeSequence(list):
    def __init__(
        self,
        token_ids,
        *,
        block_table,
        block_size=256,
    ):
        super().__init__(token_ids)
        self.step_is_decode = True
        self.step_do_sample = True
        self.block_table = block_table
        self.block_size = block_size
        self.last_block_num_tokens = (
            (len(token_ids) - 1) % block_size
        ) + 1
        self.num_blocks = len(block_table)

    @property
    def last_token(self):
        return self[-1]


def test_prepare_decode_uses_appended_last_token_zero_based_position():
    runner = make_runner()
    sequence = DecodeSequence(
        [10, 11, 12],
        block_table=[7],
    )

    input_ids, positions = runner.prepare_decode([sequence])

    assert input_ids.values == [12]
    assert positions.values == [2]


def test_prepare_mixed_decode_row_uses_appended_last_token_position():
    runner = make_runner()
    sequence = DecodeSequence(
        [20, 21, 22, 23],
        block_table=[8],
    )

    input_ids, positions = runner.prepare_mixed([sequence])

    assert input_ids.values == [23]
    assert positions.values == [3]


def test_prepare_mixed_preserves_prefill_positions_and_decode_position():
    runner = make_runner()
    prefill = PrefillSequence(
        [30, 31, 32, 33],
        chunk_start=1,
        chunk_end=3,
        block_table=[9],
    )
    prefill.step_is_decode = False
    prefill.step_do_sample = False
    decode = DecodeSequence(
        [40, 41, 42],
        block_table=[10],
    )

    input_ids, positions = runner.prepare_mixed([prefill, decode])

    assert input_ids.values == [31, 32, 42]
    assert positions.values == [1, 2, 2]


def test_prepare_prefill_installs_full_prompt_attention_reference_lengths():
    runner = make_runner()
    runner.config.kv_offload_blockwise_blocks = 1
    runner.config.am_compact_selector = "highest"
    runner.config.am_compact_score_method = "rms"
    runner.config.am_compact_beta_bound = 3.0
    runner.config.am_compact_ridge_lambda = 1e-6
    runner.config.am_omp_candidate_pool_size = 0
    runner.config.am_compact_cache_refresh_interval = 0
    runner.config.am_prefill_cache_ref_query_stride = 8
    runner.config.am_compact_num_clusters = 1
    runner.config.am_compact_route_top_k = 1
    runner.config.am_compact_num_key_spans = 1
    runner.config.am_compact_decode_refit = False
    runner.config.am_compact_decode_refit_mode = "full"
    runner.config.am_compact_decode_refit_interval = 1
    runner.config.am_compact_skip_first_layers = 0
    runner.config.am_compact_skip_last_layers = 0
    runner.config.am_compact_enable_layers = None
    runner.config.am_compact_layer_stride = 1
    runner._kv_offload_translate_slots_for_positions = (
        lambda block_table, positions, **_kwargs: [
            block_table[position // runner.block_size] * runner.block_size
            + position % runner.block_size
            for position in positions
        ]
    )
    runner._kv_offload_mark_pending_dirty = lambda *_args: None
    seqs = [
        PrefillSequence(
            [10, 11, 12, 13],
            chunk_start=0,
            chunk_end=2,
            block_table=[3],
        ),
        PrefillSequence(
            [20, 21],
            chunk_start=0,
            chunk_end=2,
            block_table=[4],
        ),
    ]

    input_ids, positions = runner.prepare_prefill(seqs)

    current = context.get_context()
    assert input_ids.values == [10, 11, 20, 21]
    assert positions.values == [0, 1, 0, 1]
    assert current.prefill_attention_reference_lens == (4, 2)


def test_prepare_spec_verify_installs_reference_context():
    runner = make_runner()
    input_ids, positions, metadata = runner.prepare_spec_verify(
        SimpleNamespace(block_size=256),
        input_tokens=[10, 20, 30],
        proxy_block_table=[0],
        slot_positions=[52, 53, 54],
    )

    current = context.get_context()
    assert input_ids.values == [10, 20, 30]
    assert positions.values == [52, 53, 54]
    assert metadata.query_len == 3
    assert metadata.logical_slots == (52, 53, 54)
    assert metadata.physical_slots == (52, 53, 54)
    assert metadata.context_len == 55
    assert current.mode == "spec_verify"
    assert current.context_lens.values == [55]
    assert current.block_tables.values == [[0]]
    assert current.flash_attn_num_splits == 16


def _batch_item(
    sequence_id,
    *,
    input_tokens,
    positions,
    logical_slots,
    context_len,
    visible_block_count,
    proxy_block_table,
    original_block_identities=(),
    reserved_block_identities=(),
    transaction_authorization=None,
):
    return SimpleNamespace(
        sequence_id=sequence_id,
        plan=verifier.SpecVerifyPlan(
            input_tokens=tuple(input_tokens),
            positions=tuple(positions),
            logical_slots=tuple(logical_slots),
            context_len=context_len,
            visible_block_count=visible_block_count,
        ),
        proxy_block_table=tuple(proxy_block_table),
        original_block_identities=tuple(
            original_block_identities
        ),
        reserved_block_identities=tuple(
            reserved_block_identities
        ),
        transaction_authorization=transaction_authorization,
    )


def _transaction_authorization(
    *,
    sequence_id=7,
    original_num_tokens=3,
    proposed_token_count=3,
    materialized_token_count=0,
    state="reserved",
    original_block_identities=((1, 3),),
    reserved_block_identities=((2, 5),),
    authorization_sha256=None,
):
    payload = (
        sequence_id,
        original_num_tokens,
        proposed_token_count,
        materialized_token_count,
        state,
        original_block_identities,
        reserved_block_identities,
    )
    return SimpleNamespace(
        sequence_id=sequence_id,
        original_num_tokens=original_num_tokens,
        proposed_token_count=proposed_token_count,
        materialized_token_count=materialized_token_count,
        state=state,
        original_block_identities=original_block_identities,
        reserved_block_identities=reserved_block_identities,
        authorization_sha256=(
            hashlib.sha256(
                repr(payload).encode("utf-8")
            ).hexdigest()
            if authorization_sha256 is None
            else authorization_sha256
        ),
    )


def _authorized_batch_item(**authorization_overrides):
    authorization = _transaction_authorization(
        **authorization_overrides
    )
    return _batch_item(
        7,
        input_tokens=(10, 11),
        positions=(3, 4),
        logical_slots=(3, 4),
        context_len=5,
        visible_block_count=2,
        proxy_block_table=(1, 2),
        original_block_identities=((1, 3),),
        reserved_block_identities=((2, 5),),
        transaction_authorization=authorization,
    )


def test_graph_enabled_prepare_spec_verify_requires_transaction_authorization():
    runner = make_runner(spec_verify_cuda_graphs=True)
    runner.block_size = 4
    item = _authorized_batch_item()
    item.transaction_authorization = None

    with pytest.raises(
        (RuntimeError, ValueError),
        match="authorization",
    ):
        runner.prepare_spec_verify_batch((item,))


@pytest.mark.parametrize(
    "authorization_overrides",
    (
        {"sequence_id": 8},
        {"state": "materialized"},
        {"materialized_token_count": 1},
        {"proposed_token_count": 2},
        {"original_block_identities": ((9, 3),)},
        {"reserved_block_identities": ((8, 5),)},
        {"authorization_sha256": "tampered"},
    ),
)
def test_graph_enabled_prepare_spec_verify_rejects_tampered_authorization(
    authorization_overrides,
):
    runner = make_runner(spec_verify_cuda_graphs=True)
    runner.block_size = 4
    item = _authorized_batch_item(
        **authorization_overrides
    )

    with pytest.raises(
        (RuntimeError, ValueError),
        match="authorization",
    ):
        runner.prepare_spec_verify_batch((item,))


def test_graph_enabled_prepare_spec_verify_accepts_exact_authorization():
    runner = make_runner(spec_verify_cuda_graphs=True)
    runner.block_size = 4

    _, _, metadata = runner.prepare_spec_verify_batch(
        (_authorized_batch_item(),)
    )

    assert metadata.rows[0].physical_slots == (7, 8)


def test_prepare_spec_verify_batch_flattens_homogeneous_rows_once():
    runner = make_runner()
    runner.block_size = 4
    uploads = []
    runner._list_to_cuda = lambda data, name, dtype: (
        uploads.append((name, list(data), dtype))
        or FakeTensor(list(data))
    )
    block_uploads = []
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables": (
            block_uploads.append(
                (name, [list(row) for row in rows])
            )
            or FakeTensor([list(row) for row in rows])
        )
    )
    items = (
        _batch_item(
            8,
            input_tokens=(10, 11),
            positions=(4, 5),
            logical_slots=(4, 5),
            context_len=6,
            visible_block_count=2,
            proxy_block_table=(5, 6),
        ),
        _batch_item(
            4,
            input_tokens=(20, 21),
            positions=(8, 9),
            logical_slots=(8, 9),
            context_len=10,
            visible_block_count=3,
            proxy_block_table=(10, 11, 12),
        ),
    )

    input_ids, positions, metadata = (
        runner.prepare_spec_verify_batch(items)
    )

    assert input_ids.values == [10, 11, 20, 21]
    assert positions.values == [4, 5, 8, 9]
    assert [row.sequence_id for row in metadata.rows] == [8, 4]
    assert [row.query_offset for row in metadata.rows] == [0, 2]
    assert metadata.query_len == 2
    assert metadata.total_query_tokens == 4
    assert metadata.block_table_width == 3
    assert metadata.rows[0].physical_slots == (24, 25)
    assert metadata.rows[1].physical_slots == (48, 49)
    assert uploads == [
        ("spec_verify_input_ids", [10, 11, 20, 21], "int64"),
        ("spec_verify_positions", [4, 5, 8, 9], "int64"),
        (
            "spec_verify_slot_mapping",
            [24, 25, 48, 49],
            "int32",
        ),
        ("spec_verify_context_lens", [6, 10], "int32"),
    ]
    assert block_uploads == [
        (
            "spec_verify_block_tables",
            [[5, 6, -1], [10, 11, 12]],
        ),
    ]
    current = context.get_context()
    assert current.mode == "spec_verify"
    assert current.context_lens.values == [6, 10]
    assert current.block_tables.values == [
        [5, 6, -1],
        [10, 11, 12],
    ]
    assert current.spec_verify_query_lens == (2, 2)


def test_offload_spec_verify_requires_prepared_residency_ticket():
    runner = make_runner(kv_offload_mvp0=True)
    runner.block_size = 4
    runner.kv_offload = SimpleNamespace()
    runner.speculative_residency = SimpleNamespace(
        is_prepared_for=lambda *args: False,
    )
    runner._list_to_cuda = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("upload must not run"))
    item = _batch_item(
        8,
        input_tokens=(10, 11),
        positions=(4, 5),
        logical_slots=(4, 5),
        context_len=6,
        visible_block_count=2,
        proxy_block_table=(5, 6),
    )

    with pytest.raises(RuntimeError, match="residency ticket"):
        runner.prepare_spec_verify_batch((item,))


def test_offload_spec_verify_maps_ticket_blocks_and_slots_to_physical():
    runner = make_runner(kv_offload_mvp0=True)
    runner.block_size = 4
    manager = SimpleNamespace(
        logical_to_slot={5: 1, 6: 3},
        map_block_rows=(
            lambda rows:
            [
                [
                    {5: 1, 6: 3}[block_id]
                    for block_id in row
                ]
                for row in rows
            ]
        ),
        map_slots_for_positions=(
            lambda block_table, positions:
            [
                {5: 1, 6: 3}[
                    block_table[position // 4]
                ] * 4
                + position % 4
                for position in positions
            ]
        ),
    )
    prepared_checks = []
    runner.kv_offload = manager
    runner.speculative_residency = SimpleNamespace(
        manager=manager,
        is_prepared_for=(
            lambda ticket_id, sequence_ids:
            prepared_checks.append(
                (ticket_id, sequence_ids)
            )
            or True
        ),
    )
    block_uploads = []
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables":
        block_uploads.append(
            (name, [list(row) for row in rows])
        )
        or FakeTensor([list(row) for row in rows])
    )
    item = _batch_item(
        8,
        input_tokens=(10, 11),
        positions=(4, 5),
        logical_slots=(4, 5),
        context_len=6,
        visible_block_count=2,
        proxy_block_table=(5, 6),
    )

    _, _, metadata = runner.prepare_spec_verify_batch(
        (item,),
        residency_ticket_id=41,
    )

    assert prepared_checks == [(41, (8,))]
    assert metadata.rows[0].block_table == (1, 3)
    assert metadata.rows[0].physical_slots == (12, 13)
    assert block_uploads == [
        ("spec_verify_block_tables", [[1, 3]]),
    ]


def test_blockwise_offload_spec_verify_keeps_logical_rows_and_maps_only_writes():
    runner = make_runner(
        kv_offload_mvp0=True,
        kv_offload_blockwise_decode=True,
        kv_offload_blockwise_prefill=True,
        kv_offload_blockwise_blocks=2,
    )
    runner.block_size = 4
    map_slots_calls = []

    def map_slots_for_positions(block_table, positions):
        map_slots_calls.append(
            (list(block_table), list(positions))
        )
        return [
            2 * 4 + position % 4
            for position in positions
        ]

    manager = SimpleNamespace(
        logical_to_slot={8: 2},
        map_block_rows=lambda _rows: (
            _ for _ in ()
        ).throw(
            AssertionError(
                "blockwise verifier must not map full history"
            )
        ),
        map_slots_for_positions=map_slots_for_positions,
    )
    runner.kv_offload = manager
    runner.speculative_residency = SimpleNamespace(
        manager=manager,
        is_prepared_for=lambda ticket_id, sequence_ids: (
            ticket_id == 41 and sequence_ids == (8,)
        ),
        ensure_materialized_for=lambda *_args: None,
    )
    block_uploads = []
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables":
        block_uploads.append(
            (name, [list(row) for row in rows])
        )
        or FakeTensor([list(row) for row in rows])
    )
    item = _batch_item(
        8,
        input_tokens=(10, 11),
        positions=(14, 15),
        logical_slots=(14, 15),
        context_len=16,
        visible_block_count=4,
        proxy_block_table=(5, 6, 7, 8),
    )

    _, _, metadata = runner.prepare_spec_verify_batch(
        (item,),
        residency_ticket_id=41,
    )

    assert map_slots_calls == [
        ([5, 6, 7, 8], [14, 15]),
    ]
    assert metadata.rows[0].block_table == (5, 6, 7, 8)
    assert metadata.rows[0].physical_slots == (10, 11)
    assert block_uploads == [
        (
            "spec_verify_block_tables",
            [[5, 6, 7, 8]],
        ),
    ]
    current = context.get_context()
    assert current.kv_offload_manager is manager
    assert current.kv_offload_blockwise_decode is True
    assert current.kv_offload_blockwise_blocks == 2
    assert current.kv_offload_logical_block_tables == [
        [5, 6, 7, 8],
    ]
    assert current.kv_offload_context_lens == [16]
    assert current.kv_offload_write_blocks == [8]
    assert current.spec_verify_query_lens == (2,)


def test_blockwise_offload_revalidates_ticket_residency_before_mapping():
    runner = make_runner(
        kv_offload_mvp0=True,
        kv_offload_blockwise_decode=True,
        kv_offload_blockwise_prefill=True,
        kv_offload_blockwise_blocks=2,
    )
    runner.block_size = 4
    manager = SimpleNamespace(logical_to_slot={})
    calls = []

    def ensure_materialized_for(ticket_id, sequence_ids):
        calls.append((ticket_id, sequence_ids))
        manager.logical_to_slot[8] = 2

    def map_slots_for_positions(block_table, positions):
        return [
            manager.logical_to_slot[
                block_table[position // 4]
            ]
            * 4
            + position % 4
            for position in positions
        ]

    manager.map_slots_for_positions = map_slots_for_positions
    manager.map_block_rows = lambda _rows: (
        _ for _ in ()
    ).throw(
        AssertionError(
            "blockwise verifier must not map full history"
        )
    )
    runner.kv_offload = manager
    runner.speculative_residency = SimpleNamespace(
        manager=manager,
        is_prepared_for=lambda ticket_id, sequence_ids: (
            ticket_id == 41 and sequence_ids == (8,)
        ),
        ensure_materialized_for=ensure_materialized_for,
    )
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables": FakeTensor(rows)
    )
    item = _batch_item(
        8,
        input_tokens=(10, 11),
        positions=(14, 15),
        logical_slots=(14, 15),
        context_len=16,
        visible_block_count=4,
        proxy_block_table=(5, 6, 7, 8),
    )

    _, _, metadata = runner.prepare_spec_verify_batch(
        (item,),
        residency_ticket_id=41,
    )

    assert calls == [(41, (8,))]
    assert metadata.rows[0].physical_slots == (10, 11)


def test_offload_spec_verify_marks_ticket_after_forward():
    runner = make_runner(kv_offload_mvp0=True)
    metadata = _batch_metadata()
    runner.prepare_spec_verify_batch = (
        lambda items, residency_ticket_id=None: (
            FakeTensor([10, 11, 20, 21]),
            FakeTensor([5, 6, 9, 10]),
            metadata,
        )
    )
    materialized = []
    runner.speculative_residency = SimpleNamespace(
        mark_materialized=(
            lambda ticket_id, sequence_ids:
            materialized.append(
                (ticket_id, sequence_ids)
            )
        )
    )
    runner.run_model = lambda *args, **kwargs: (
        _FakeGreedyLogits([101, 102, 201, 202])
    )

    runner.run_spec_verify_batch(
        (
            SimpleNamespace(sequence_id=8),
            SimpleNamespace(sequence_id=4),
        ),
        residency_ticket_id=41,
    )

    assert materialized == [(41, (8, 4))]


@pytest.mark.parametrize(
    "items,match",
    [
        ((), "non-empty"),
        (
            (
                _batch_item(
                    8,
                    input_tokens=(10,),
                    positions=(4,),
                    logical_slots=(4,),
                    context_len=5,
                    visible_block_count=2,
                    proxy_block_table=(5, 6),
                ),
                _batch_item(
                    8,
                    input_tokens=(20,),
                    positions=(8,),
                    logical_slots=(8,),
                    context_len=9,
                    visible_block_count=3,
                    proxy_block_table=(10, 11, 12),
                ),
            ),
            "unique",
        ),
        (
            (
                _batch_item(
                    8,
                    input_tokens=(),
                    positions=(),
                    logical_slots=(),
                    context_len=4,
                    visible_block_count=1,
                    proxy_block_table=(5,),
                ),
            ),
            "query",
        ),
        (
            (
                _batch_item(
                    8,
                    input_tokens=(10,),
                    positions=(4,),
                    logical_slots=(4,),
                    context_len=5,
                    visible_block_count=2,
                    proxy_block_table=(5, 6),
                ),
                _batch_item(
                    4,
                    input_tokens=(20, 21),
                    positions=(8, 9),
                    logical_slots=(8, 9),
                    context_len=10,
                    visible_block_count=3,
                    proxy_block_table=(10, 11, 12),
                ),
            ),
            "homogeneous",
        ),
        (
            (
                SimpleNamespace(
                    sequence_id=8,
                    plan=object(),
                    proxy_block_table=(5,),
                ),
            ),
            "SpecVerifyPlan",
        ),
        (
            (
                _batch_item(
                    8,
                    input_tokens=(10,),
                    positions=(4,),
                    logical_slots=(4,),
                    context_len=5,
                    visible_block_count=2,
                    proxy_block_table=(5,),
                ),
            ),
            "cover",
        ),
        (
            (
                _batch_item(
                    8,
                    input_tokens=(10,),
                    positions=(4,),
                    logical_slots=(4,),
                    context_len=5,
                    visible_block_count=2,
                    proxy_block_table=(5, -1),
                ),
            ),
            "invalid block",
        ),
    ],
)
def test_prepare_spec_verify_batch_rejects_before_upload(
    items,
    match,
):
    runner = make_runner()
    runner.block_size = 4
    runner._list_to_cuda = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("upload must not run"))

    with pytest.raises((ValueError, RuntimeError), match=match):
        runner.prepare_spec_verify_batch(items)


def test_step_logits_recording_accessor_is_default_off_and_returns_clone():
    runner = make_runner()
    runner._record_step_logits = False
    runner._last_step_logits_cpu = None

    assert runner.last_step_logits() is None
    runner.enable_step_logits_recording(True)
    assert runner._record_step_logits is True
    assert runner.last_step_logits() is None

    stored = FakeIndexedTensor([[1.0, 2.0]])
    runner._last_step_logits_cpu = stored
    returned = runner.last_step_logits()
    assert returned is not stored
    assert returned.values == ("cloned", stored.values)

    runner.enable_step_logits_recording(False)
    assert runner._record_step_logits is False
    assert runner.last_step_logits() is None


def test_snapshot_kv_slots_uses_physical_block_and_offset_indices():
    runner = make_runner()
    runner.block_size = 4
    runner.kv_cache = FakeIndexedTensor("kv-cache")
    runner.kv_cache.device = "cuda:0"

    snapshot = runner.snapshot_kv_slots([3, 4, 9])

    assert set(snapshot) == {"keys", "values"}
    key_index = runner.kv_cache.trace[0][1]
    value_index = runner.kv_cache.trace[4][1]
    assert key_index[0] == 0
    assert key_index[1] == slice(None)
    assert key_index[2].values["values"] == [0, 1, 2]
    assert key_index[3].values["values"] == [3, 0, 1]
    assert value_index[0] == 1
    assert value_index[1] == slice(None)
    assert value_index[2].values["values"] == [0, 1, 2]
    assert value_index[3].values["values"] == [3, 0, 1]
    assert runner.kv_cache.trace.count(("detach", None)) == 2
    assert runner.kv_cache.trace.count(("cpu", None)) == 2
    assert runner.kv_cache.trace.count(("clone", None)) == 2


def test_snapshot_kv_slots_rejects_empty_or_quantized_requests():
    runner = make_runner()
    runner.kv_cache = FakeIndexedTensor("kv-cache")
    runner.kv_cache.device = "cuda:0"

    try:
        runner.snapshot_kv_slots([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty snapshot request must fail")

    runner = make_runner(kv_quant_bits=4)
    runner.kv_cache = FakeIndexedTensor("kv-cache")
    runner.kv_cache.device = "cuda:0"
    try:
        runner.snapshot_kv_slots([0])
    except RuntimeError as exc:
        assert "FP KV" in str(exc)
    else:
        raise AssertionError("quantized KV snapshot must fail")


def test_prepare_spec_verify_rejects_nonconsecutive_slots_before_upload():
    runner = make_runner()
    runner._list_to_cuda = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("upload must not run"))

    try:
        runner.prepare_spec_verify(
            SimpleNamespace(block_size=256),
            input_tokens=[10, 20],
            proxy_block_table=[0],
            slot_positions=[52, 54],
        )
    except ValueError as exc:
        assert "consecutive" in str(exc)
    else:
        raise AssertionError("nonconsecutive slots must fail")


def test_every_unsupported_feature_fails_closed():
    unsupported = {
        "kv_quant_bits": 4,
        "kv_offload_mvp0": True,
        "quest_top_k_blocks": 1,
        "am_compact_blocks": 1,
        "kv_cartridge_blocks": 1,
        "chunked_prefill_mixed_batch": True,
    }

    for name, value in unsupported.items():
        runner = make_runner(**{name: value})
        try:
            runner._validate_spec_verify_compatibility(
                seq_count=1,
                linear_draft=True,
                greedy=True,
                mixed_batch=False,
            )
        except RuntimeError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(name)


def test_blockwise_spec_verify_is_allowed_with_kv_offload():
    runner = make_runner(
        kv_offload_mvp0=True,
        kv_offload_blockwise_decode=True,
        kv_offload_blockwise_prefill=True,
    )
    runner.speculative_residency = SimpleNamespace(
        is_prepared_for=lambda ticket_id, sequence_ids: (
            ticket_id == 41 and sequence_ids == (7,)
        ),
    )

    runner._validate_spec_verify_compatibility(
        seq_count=1,
        linear_draft=True,
        greedy=True,
        mixed_batch=False,
        residency_ticket_id=41,
        sequence_ids=(7,),
    )


def test_blockwise_spec_verify_requires_kv_offload():
    runner = make_runner(
        kv_offload_mvp0=False,
        kv_offload_blockwise_decode=True,
    )

    with pytest.raises(RuntimeError, match="kv_offload_mvp0"):
        runner._validate_spec_verify_compatibility(
            seq_count=1,
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
            require_residency_ticket=False,
        )


def test_multi_sequence_is_allowed_but_invalid_modes_fail():
    runner = make_runner()
    runner._validate_spec_verify_compatibility(
        seq_count=2,
        linear_draft=True,
        greedy=True,
        mixed_batch=False,
    )
    invalid = (
        dict(
            seq_count=0,
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
        ),
        dict(
            seq_count=1,
            linear_draft=False,
            greedy=True,
            mixed_batch=False,
        ),
        dict(
            seq_count=1,
            linear_draft=True,
            greedy=False,
            mixed_batch=False,
        ),
        dict(
            seq_count=1,
            linear_draft=True,
            greedy=True,
            mixed_batch=True,
        ),
    )

    for arguments in invalid:
        try:
            runner._validate_spec_verify_compatibility(**arguments)
        except RuntimeError:
            pass
        else:
            raise AssertionError(arguments)


def test_spec_verify_run_model_uses_eager_and_keeps_all_rows():
    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            return FakeTensor([[1], [2], [3]])

        def compute_logits(self, hidden):
            return hidden

    runner = make_runner()
    runner.model = FakeModel()
    runner.graphs = {"forbidden": True}
    runner.graph_bs = []
    runner.graph_vars = {}

    logits = runner.run_model(
        FakeTensor([10, 20, 30]),
        FakeTensor([53, 54, 55]),
        is_prefill=False,
        execution_mode="spec_verify",
    )

    assert logits.values == [[1], [2], [3]]


def _batch_metadata():
    return verifier.SpecVerifyBatchMetadata(
        rows=(
            verifier.SpecVerifyBatchRowMetadata(
                sequence_id=8,
                batch_index=0,
                query_offset=0,
                query_len=2,
                input_tokens=(10, 11),
                positions=(5, 6),
                logical_slots=(4, 5),
                physical_slots=(24, 25),
                context_len=6,
                block_table=(5, 6),
            ),
            verifier.SpecVerifyBatchRowMetadata(
                sequence_id=4,
                batch_index=1,
                query_offset=2,
                query_len=2,
                input_tokens=(20, 21),
                positions=(9, 10),
                logical_slots=(8, 9),
                physical_slots=(48, 49),
                context_len=10,
                block_table=(10, 11, 12),
            ),
        ),
        query_len=2,
        total_query_tokens=4,
        block_table_width=3,
    )


def _profile_request_set_sha256(sequence_ids):
    return hashlib.sha256(
        json.dumps(
            sorted(int(value) for value in sequence_ids),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _capture_profile_steps():
    steps = []
    original = model_runner.run_profiled_step

    def capture(_profiler, **kwargs):
        steps.append({
            name: kwargs[name]
            for name in (
                "batch_kind",
                "is_decode",
                "active_sequence_count",
                "request_set_sha256",
                "dispatch",
            )
        })
        return kwargs["call"]()

    model_runner.run_profiled_step = capture
    return steps, original


def _profiled_first_target_runner(rank):
    runner = make_runner()
    runner.rank = rank
    runner.prepare_decode = lambda seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner.run_model = lambda *args, **kwargs: (
        _FakeGreedyLogits([101, 201])
    )
    return runner


def test_run_spec_first_target_batch_profiles_rank_zero_callback():
    runner = _profiled_first_target_runner(rank=0)
    seqs = _first_target_sequences(0, 0)
    steps, original = _capture_profile_steps()
    try:
        rows = runner.run_spec_first_target_batch(seqs)
    finally:
        model_runner.run_profiled_step = original

    assert rows is not None
    assert steps == [{
        "batch_kind": "spec_first_target",
        "is_decode": True,
        "active_sequence_count": 2,
        "request_set_sha256": _profile_request_set_sha256(
            (8, 4)
        ),
        "dispatch": "eager",
    }]


def test_run_spec_first_target_batch_profiles_worker_before_none_result():
    runner = _profiled_first_target_runner(rank=1)
    seqs = _first_target_sequences(0, 0)
    steps, original = _capture_profile_steps()
    try:
        rows = runner.run_spec_first_target_batch(seqs)
    finally:
        model_runner.run_profiled_step = original

    assert rows is None
    assert steps == [{
        "batch_kind": "spec_first_target",
        "is_decode": True,
        "active_sequence_count": 2,
        "request_set_sha256": _profile_request_set_sha256(
            (8, 4)
        ),
        "dispatch": "eager",
    }]


def _profiled_verify_runner(rank):
    runner = make_runner()
    runner.rank = rank
    metadata = _batch_metadata()
    runner.prepare_spec_verify_batch = lambda items: (
        FakeTensor([10, 11, 20, 21]),
        FakeTensor([5, 6, 9, 10]),
        metadata,
    )
    runner.run_model = lambda *args, **kwargs: (
        _FakeGreedyLogits([101, 102, 201, 202])
    )
    return runner


def _profiled_tail_items():
    return (
        SimpleNamespace(sequence_id=8),
        SimpleNamespace(sequence_id=4),
    )


def test_run_spec_verify_batch_profiles_rank_zero_callback():
    runner = _profiled_verify_runner(rank=0)
    items = _profiled_tail_items()
    steps, original = _capture_profile_steps()
    try:
        rows = runner.run_spec_verify_batch(items)
    finally:
        model_runner.run_profiled_step = original

    assert rows is not None
    assert steps == [{
        "batch_kind": "spec_verify",
        "is_decode": True,
        "active_sequence_count": 2,
        "request_set_sha256": _profile_request_set_sha256(
            (8, 4)
        ),
        "dispatch": "eager",
    }]


def test_run_spec_verify_batch_profiles_worker_before_none_result():
    runner = _profiled_verify_runner(rank=2)
    items = _profiled_tail_items()
    steps, original = _capture_profile_steps()
    try:
        rows = runner.run_spec_verify_batch(items)
    finally:
        model_runner.run_profiled_step = original

    assert rows is None
    assert steps == [{
        "batch_kind": "spec_verify",
        "is_decode": True,
        "active_sequence_count": 2,
        "request_set_sha256": _profile_request_set_sha256(
            (8, 4)
        ),
        "dispatch": "eager",
    }]


class _FakeGreedyLogits:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.argmax_calls = []
        self.index_calls = []

    def argmax(self, dim):
        self.argmax_calls.append(dim)
        return SimpleNamespace(
            tolist=lambda: list(self.token_ids)
        )

    def __getitem__(self, index):
        self.index_calls.append(index)
        return ("logits", index, self.token_ids[index])


def _fake_tp_greedy_selector(
    logits,
    *,
    rank,
    world_size,
    batch_size,
    device,
):
    del device
    assert rank == 0
    assert world_size == 1
    assert batch_size == len(logits.token_ids)
    return SimpleNamespace(
        tolist=lambda: logits.argmax(dim=-1).tolist()
    )


model_runner.select_tensor_parallel_greedy_tokens = (
    _fake_tp_greedy_selector
)


def test_run_spec_verify_batch_uses_one_forward_and_splits_rows():
    runner = make_runner()
    calls = []
    metadata = _batch_metadata()
    runner.prepare_spec_verify_batch = lambda items: (
        calls.append(("prepare", items))
        or (
            FakeTensor([10, 11, 20, 21]),
            FakeTensor([5, 6, 9, 10]),
            metadata,
        )
    )
    logits = _FakeGreedyLogits([101, 102, 201, 202])
    runner.run_model = lambda *args, **kwargs: (
        calls.append(("run_model", args, kwargs))
        or logits
    )
    reset_calls = []
    original_reset = model_runner.reset_context
    model_runner.reset_context = lambda: reset_calls.append("reset")
    items = _profiled_tail_items()
    try:
        rows = runner.run_spec_verify_batch(items)
    finally:
        model_runner.reset_context = original_reset

    assert rows == (
        verifier.SpecVerifyBatchResultRow(
            sequence_id=8,
            target_tokens=(101, 102),
        ),
        verifier.SpecVerifyBatchResultRow(
            sequence_id=4,
            target_tokens=(201, 202),
        ),
    )
    assert calls[0] == ("prepare", items)
    assert calls[1][0] == "run_model"
    assert calls[1][1][2] is False
    assert calls[1][2] == {"execution_mode": "spec_verify"}
    assert logits.argmax_calls == [-1]
    assert reset_calls == ["reset"]


def test_run_spec_verify_batch_records_tail_trace_without_changing_tokens():
    runner = _trace_ready_runner()
    runner.enable_spec_verify_trace_recording(True)
    runner.set_spec_verify_trace_context(
        "native_mtp",
        2,
        5,
    )
    runner.kv_offload = SimpleNamespace(
        bound_generations=[1] * 129,
    )
    block_table = tuple(range(129))
    metadata = verifier.SpecVerifyBatchMetadata(
        rows=(
            verifier.SpecVerifyBatchRowMetadata(
                sequence_id=7,
                batch_index=0,
                query_offset=0,
                query_len=3,
                input_tokens=(15, 15, 2658),
                positions=(32768, 32769, 32770),
                logical_slots=(32768, 32769, 32770),
                physical_slots=(32768, 32769, 32770),
                context_len=32771,
                block_table=block_table,
            ),
            verifier.SpecVerifyBatchRowMetadata(
                sequence_id=9,
                batch_index=1,
                query_offset=3,
                query_len=3,
                input_tokens=(31, 32, 33),
                positions=(32768, 32769, 32770),
                logical_slots=(32768, 32769, 32770),
                physical_slots=(32768, 32769, 32770),
                context_len=32771,
                block_table=block_table,
            ),
        ),
        query_len=3,
        total_query_tokens=6,
        block_table_width=129,
    )
    runner.prepare_spec_verify_batch = lambda _items: (
        _TraceTensor([15, 15, 2658, 31, 32, 33]),
        _TraceTensor(
            [32768, 32769, 32770, 32768, 32769, 32770]
        ),
        metadata,
    )
    logits = _TraceTensor([
        [0.0, 9.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 9.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 9.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 9.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 9.0],
        [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ])
    runner.run_model = lambda *_args, **_kwargs: logits
    items = (
        SimpleNamespace(
            sequence_id=7,
            sequence=SimpleNamespace(
                seq_id=7,
                num_completion_tokens=0,
            ),
        ),
        SimpleNamespace(
            sequence_id=9,
            sequence=SimpleNamespace(
                seq_id=9,
                num_completion_tokens=4,
            ),
        ),
    )

    results = runner.run_spec_verify_batch(items)
    rows = runner.drain_spec_verify_trace_rows()

    assert results == (
        verifier.SpecVerifyBatchResultRow(
            sequence_id=7,
            target_tokens=(1, 2, 3),
        ),
        verifier.SpecVerifyBatchResultRow(
            sequence_id=9,
            target_tokens=(4, 5, 0),
        ),
    )
    assert [row["stage"] for row in rows] == [
        "verify_tail",
    ] * 6
    assert [
        row["prediction_index"] for row in rows[:3]
    ] == [1, 2, 3]
    assert [row["input_token_id"] for row in rows[:3]] == [
        15,
        15,
        2658,
    ]
    assert [row["position"] for row in rows[:3]] == [
        32768,
        32769,
        32770,
    ]
    assert rows[0]["query_offset"] == 0
    assert rows[3]["query_offset"] == 3


class _CandidateStateOwner:
    def __init__(self, prepared_handle):
        self.active = True
        self.prepared_handle = prepared_handle
        self.events = []

    def record_first_target(self, prepared_step):
        self.events.append(("first", prepared_step))
        return {"status": "recorded"}

    def initial_tail_candidates(self, sequence_ids):
        self.events.append(("initial_tail", sequence_ids))
        return "initial-candidates"

    def record_tail(self, prepared_step, sequence_ids):
        self.events.append(
            ("tail", prepared_step, sequence_ids)
        )
        return {"status": "recorded"}


def test_candidate_state_first_target_records_without_live_commit():
    runner = make_runner()
    seqs = tuple(
        SimpleNamespace(
            seq_id=sequence_id,
            temperature=0,
            hybrid_state_slot_id=slot_id,
            hybrid_state_generation=1,
        )
        for slot_id, sequence_id in enumerate((8, 4))
    )
    runner._prepare_hybrid_state_batch = lambda *_args: None
    runner.prepare_decode = lambda _seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    prepared = SimpleNamespace(
        logits=_FakeGreedyLogits([101, 201]),
        normalized=("hidden-8", "hidden-4"),
    )
    owner = _CandidateStateOwner(prepared)
    runner.qwen35_speculative_state_owner = owner
    calls = []

    def run_model(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["prepare_qwen35_state"] is True
        return prepared

    runner.run_model = run_model

    rows = runner.run_spec_first_target_batch(
        seqs,
        return_hidden=True,
    )

    assert tuple(row.target_token for row in rows) == (101, 201)
    assert tuple(row.target_hidden for row in rows) == (
        "hidden-8",
        "hidden-4",
    )
    assert owner.events == [("first", prepared)]
    assert len(calls) == 1


def test_fused_candidate_state_first_target_records_checkpoint():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities
            self.inputs = ()

        def propose_batch(self, inputs):
            self.inputs = inputs
            return tuple(
                DraftProposal(
                    sequence_id=row.sequence_id,
                    token_ids=(),
                    source_type="fixture",
                )
                for row in inputs
            )

    executor = Executor()
    runner = make_runner()
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        executor,
        capabilities,
    )
    seqs = _proposal_sequences()
    for slot_id, seq in enumerate(seqs):
        seq.hybrid_state_slot_id = slot_id
        seq.hybrid_state_generation = 1
    runner._prepare_hybrid_state_batch = lambda *_args: None
    runner.prepare_decode = lambda _seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    prepared = SimpleNamespace(
        logits=_FakeGreedyLogits([101, 201]),
        normalized=("hidden-8", "hidden-4"),
    )
    owner = _CandidateStateOwner(prepared)
    runner.qwen35_speculative_state_owner = owner
    calls = []

    def run_model(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs == {
            "return_hidden": True,
            "execution_mode": "decode",
            "prepare_qwen35_state": True,
        }
        return prepared

    runner.run_model = run_model
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )

    rows = runner.run_spec_first_target_and_proposal_batch(
        seqs,
        descriptor,
        (),
    )

    assert tuple(row.target_token for row in rows) == (101, 201)
    assert owner.events == [("first", prepared)]
    assert executor.inputs[0].target_hidden == ("hidden-8",)
    assert executor.inputs[1].target_hidden == ("hidden-4",)
    assert len(calls) == 1


def test_fused_candidate_state_first_target_records_rank_zero_logits():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities

        def propose_batch(self, inputs):
            return tuple(
                DraftProposal(
                    sequence_id=row.sequence_id,
                    token_ids=(),
                    source_type="fixture",
                )
                for row in inputs
            )

    class RecordedLogits(_FakeGreedyLogits):
        def __init__(self, token_ids):
            super().__init__(token_ids)
            self.trace = []

        def detach(self):
            self.trace.append("detach")
            return self

        def float(self):
            self.trace.append("float")
            return self

        def cpu(self):
            self.trace.append("cpu")
            return self

        def clone(self):
            self.trace.append("clone")
            return SimpleNamespace(token_ids=list(self.token_ids))

    executor = Executor()
    runner = make_runner()
    runner._record_step_logits = True
    runner._last_step_logits_cpu = None
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        executor,
        capabilities,
    )
    seqs = _proposal_sequences()
    for slot_id, seq in enumerate(seqs):
        seq.hybrid_state_slot_id = slot_id
        seq.hybrid_state_generation = 1
    runner._prepare_hybrid_state_batch = lambda *_args: None
    runner.prepare_decode = lambda _seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = RecordedLogits([101, 201])
    prepared = SimpleNamespace(
        logits=logits,
        normalized=("hidden-8", "hidden-4"),
    )
    runner.qwen35_speculative_state_owner = (
        _CandidateStateOwner(prepared)
    )
    runner.run_model = lambda *_args, **_kwargs: prepared
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )

    rows = runner.run_spec_first_target_and_proposal_batch(
        seqs,
        descriptor,
        (),
    )

    assert tuple(row.target_token for row in rows) == (101, 201)
    recorded = runner.last_step_logits()
    assert recorded.token_ids == [101, 201]
    assert logits.trace == [
        "detach",
        "float",
        "cpu",
        "clone",
    ]


def test_candidate_state_tail_uses_first_target_state_and_records_prefixes():
    runner = make_runner()
    metadata = _batch_metadata()
    items = _profiled_tail_items()
    runner.prepare_spec_verify_batch = lambda _items: (
        FakeTensor([10, 11, 20, 21]),
        FakeTensor([5, 6, 9, 10]),
        metadata,
    )
    prepared = SimpleNamespace(
        logits=_FakeGreedyLogits([101, 102, 201, 202]),
        prefix_candidates="prefix-candidates",
    )
    owner = _CandidateStateOwner(prepared)
    runner.qwen35_speculative_state_owner = owner
    runner._speculative_side_state_leases_by_sequence = {
        8: "lease-8",
        4: "lease-4",
    }
    calls = []

    def run_model(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs == {
            "execution_mode": "spec_verify",
            "prepare_qwen35_state": True,
            "initial_qwen35_candidates": "initial-candidates",
            "capture_qwen35_prefix_states": True,
        }
        return prepared

    runner.run_model = run_model

    rows = runner.run_spec_verify_batch(items)

    assert rows == (
        verifier.SpecVerifyBatchResultRow(
            sequence_id=8,
            target_tokens=(101, 102),
        ),
        verifier.SpecVerifyBatchResultRow(
            sequence_id=4,
            target_tokens=(201, 202),
        ),
    )
    assert owner.events == [
        ("initial_tail", (8, 4)),
        ("tail", prepared, (8, 4)),
    ]
    assert runner._last_hybrid_state_leases == (
        "lease-8",
        "lease-4",
    )
    assert runner._last_hybrid_state_token_counts == (2, 2)
    assert len(calls) == 1


def test_speculative_side_state_lifecycle_delegates_tensor_free_receipts():
    runner = make_runner()
    calls = []

    class Owner:
        active = False

        def prepare(self, sequences, leases):
            self.active = True
            calls.append(("prepare", sequences, leases))
            return {
                "operation": "prepare",
                "status": "prepared",
                "transaction_id": "tx-1",
                "sequence_ids": [8, 4],
            }

        def select(self, handle, rows):
            calls.append(("select", handle, rows))
            return {"status": "selected"}

        def apply(self, handle):
            calls.append(("apply", handle))
            return {"status": "applied"}

        def seal(self, handle):
            calls.append(("seal", handle))
            self.active = False
            return {"status": "sealed"}

        def rollback(self, handle):
            calls.append(("rollback", handle))
            self.active = False
            return {"status": "rolled_back"}

    owner = Owner()
    runner.qwen35_speculative_state_owner = owner
    seqs = tuple(
        SimpleNamespace(
            seq_id=sequence_id,
            hybrid_state_slot_id=slot_id,
            hybrid_state_generation=1,
        )
        for slot_id, sequence_id in enumerate((8, 4))
    )

    handle = runner.prepare_speculative_side_state_batch(seqs)
    assert runner.speculative_side_state_available()
    assert runner.select_speculative_side_state_batch(("row",)) == {
        "status": "selected"
    }
    assert runner.apply_speculative_side_state_batch() == {
        "status": "applied"
    }
    assert runner.seal_speculative_side_state_batch() == {
        "status": "sealed"
    }
    assert calls[0][0] == "prepare"
    assert calls[1] == ("select", handle, ("row",))
    assert calls[2] == ("apply", handle)
    assert calls[3] == ("seal", handle)


def test_run_spec_verify_batch_worker_executes_forward_without_result():
    runner = make_runner()
    runner.rank = 1
    metadata = _batch_metadata()
    runner.prepare_spec_verify_batch = lambda items: (
        FakeTensor([10, 11, 20, 21]),
        FakeTensor([5, 6, 9, 10]),
        metadata,
    )
    logits = _FakeGreedyLogits([101, 102, 201, 202])
    run_calls = []
    runner.run_model = lambda *args, **kwargs: (
        run_calls.append((args, kwargs))
        or logits
    )

    assert runner.run_spec_verify_batch(
        _profiled_tail_items()
    ) is None
    assert len(run_calls) == 1
    assert logits.argmax_calls == []


def test_run_spec_verify_batch_resets_context_after_forward_failure():
    runner = make_runner()
    runner.prepare_spec_verify_batch = lambda items: (
        FakeTensor([10]),
        FakeTensor([5]),
        _batch_metadata(),
    )
    runner.run_model = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("forward failed"))
    reset_calls = []
    original_reset = model_runner.reset_context
    model_runner.reset_context = lambda: reset_calls.append("reset")
    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            runner.run_spec_verify_batch(
                (SimpleNamespace(sequence_id=8),)
            )
    finally:
        model_runner.reset_context = original_reset

    assert reset_calls == ["reset"]


def _first_target_sequences(*temperatures):
    return tuple(
        SimpleNamespace(
            seq_id=sequence_id,
            temperature=temperature,
            block_table=[sequence_id],
        )
        for sequence_id, temperature in zip(
            (8, 4),
            temperatures,
        )
    )


class _TraceTensor(FakeTensor):
    def __init__(self, values, *, device="cuda:0"):
        super().__init__(values, device=device)
        self.ndim = (
            2
            if values
            and isinstance(values[0], (list, tuple))
            else 1
        )
        self.shape = (
            (len(values), len(values[0]))
            if self.ndim == 2
            else (len(values),)
        )
        self.token_ids = (
            [
                max(
                    range(len(row)),
                    key=lambda index: row[index],
                )
                for row in values
            ]
            if self.ndim == 2
            else []
        )

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)

    def clone(self):
        return _TraceTensor(
            copy.deepcopy(self.values),
            device=self.device,
        )

    def argmax(self, dim):
        assert dim == -1
        return SimpleNamespace(
            tolist=lambda: list(self.token_ids)
        )


def _trace_first_target_sequence():
    return SimpleNamespace(
        seq_id=7,
        temperature=0,
        block_table=list(range(128)),
        num_tokens=32768,
        num_completion_tokens=0,
        max_tokens=8,
    )


def _enable_first_target_trace(runner):
    runner.enable_spec_verify_trace_recording(True)
    runner.set_spec_verify_trace_context(
        "native_mtp",
        1,
        4,
    )
    runner.kv_offload = SimpleNamespace(
        bound_generations=[1] * 128,
        bind_logical_block_identity=lambda *_args: None,
    )
    runner._prepare_hybrid_state_batch = lambda *_args: None
    runner.prepare_decode = lambda _seqs: (
        _TraceTensor([11]),
        _TraceTensor([32767]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None


def test_run_spec_first_target_batch_records_trace_without_changing_token():
    runner = _trace_ready_runner()
    _enable_first_target_trace(runner)
    logits = _TraceTensor([[0.0, 1.0, 2.0, 3.0, 9.0, 4.0]])
    runner.run_model = lambda *_args, **_kwargs: logits
    seq = _trace_first_target_sequence()

    results = runner.run_spec_first_target_batch(
        (seq,),
        kv_block_identity_rows=_identity_rows((seq,)),
    )
    rows = runner.drain_spec_verify_trace_rows()

    assert results[0].target_token == 4
    assert len(rows) == 1
    assert rows[0]["stage"] == "first_target"
    assert rows[0]["execution_mode"] == "decode"
    assert rows[0]["sequence_id"] == 7
    assert rows[0]["prediction_index"] == 0
    assert rows[0]["input_token_id"] == 11
    assert rows[0]["position"] == 32767
    assert rows[0]["context_length"] == 32768
    assert rows[0]["logical_block_identities"][-1] == (
        127,
        1,
    )


def test_fused_first_target_records_trace_without_changing_proposal():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=False,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
        requires_full_token_history=False,
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities
            self.inputs = ()

        def propose_batch(self, inputs):
            self.inputs = inputs
            return (
                DraftProposal(
                    sequence_id=7,
                    token_ids=(21, 22),
                    source_type="fixture",
                ),
            )

    runner = _trace_ready_runner()
    _enable_first_target_trace(runner)
    executor = Executor()
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        executor,
        capabilities,
    )
    logits = _TraceTensor([[0.0, 1.0, 2.0, 3.0, 9.0, 4.0]])
    runner.run_model = lambda *_args, **_kwargs: logits
    seq = _trace_first_target_sequence()
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )

    results = runner.run_spec_first_target_and_proposal_batch(
        (seq,),
        descriptor,
        _identity_rows((seq,)),
    )
    rows = runner.drain_spec_verify_trace_rows()

    assert results[0].target_token == 4
    assert results[0].proposal.token_ids == (21, 22)
    assert executor.inputs[0].first_target_token == 4
    assert len(rows) == 1
    assert rows[0]["stage"] == "first_target"
    assert rows[0]["execution_mode"] == "decode"
    assert rows[0]["sequence_id"] == 7
    assert rows[0]["prediction_index"] == 0
    assert rows[0]["input_token_id"] == 11
    assert rows[0]["position"] == 32767
    assert rows[0]["context_length"] == 32768
    assert rows[0]["logical_block_identities"][-1] == (
        127,
        1,
    )


def _identity_rows(seqs):
    return tuple(
        KVBlockIdentityRow(
            sequence_id=seq.seq_id,
            block_identities=tuple(
                (block_id, sequence_index + 1)
                for block_id in seq.block_table
            ),
        )
        for sequence_index, seq in enumerate(seqs)
    )


def _proposal_sequences():
    return (
        SimpleNamespace(
            seq_id=8,
            temperature=0,
            block_table=[8],
            token_ids=[1, 2, 8],
            num_tokens=3,
            max_tokens=8,
            num_completion_tokens=1,
        ),
        SimpleNamespace(
            seq_id=4,
            temperature=0,
            block_table=[4],
            token_ids=[1, 2, 4],
            num_tokens=3,
            max_tokens=7,
            num_completion_tokens=2,
        ),
    )


def test_run_offload_spec_first_target_and_proposal_keeps_hidden_local():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=True,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities
            self.calls = []

        def propose_batch(self, inputs):
            self.calls.append(inputs)
            return (
                DraftProposal(
                    sequence_id=4,
                    token_ids=(),
                    source_type="fixture",
                ),
                DraftProposal(
                    sequence_id=8,
                    token_ids=(11, 12),
                    source_type="fixture",
                ),
            )

    executor = Executor()
    runner = make_runner(kv_offload_mvp0=True)
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        executor,
        capabilities,
    )
    calls = []
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=(
            lambda block_id, generation:
            calls.append(("bind", block_id, generation))
        )
    )
    runner.prepare_decode = lambda seqs: (
        calls.append(("prepare_decode", tuple(seqs)))
        or (
            FakeTensor([10, 20]),
            FakeTensor([5, 9]),
        )
    )
    runner._kv_offload_before_forward = (
        lambda: calls.append(("before_forward",))
    )
    runner._kv_offload_after_forward = (
        lambda: calls.append(("after_forward",))
    )
    logits = _FakeGreedyLogits([101, 201])
    hidden_rows = ("hidden-8", "hidden-4")
    runner.run_model = lambda *args, **kwargs: (
        calls.append(("run_model", args, kwargs))
        or (logits, hidden_rows)
    )
    seqs = _proposal_sequences()
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )

    rows = runner.run_spec_first_target_and_proposal_batch(
        seqs,
        descriptor,
        _identity_rows(seqs),
    )

    assert tuple(row.sequence_id for row in rows) == (8, 4)
    assert all(
        isinstance(row, FirstTargetProposalResult)
        for row in rows
    )
    assert rows[0].target_token == 101
    assert rows[0].proposal.token_ids == (11, 12)
    assert rows[1].proposal.token_ids == ()
    assert executor.calls[0][0].target_hidden == ("hidden-8",)
    assert executor.calls[0][0].target_logits == (
        "logits",
        0,
        101,
    )
    assert not hasattr(rows[0], "target_hidden")
    assert calls[4][2] == {
        "return_hidden": True,
        "execution_mode": "decode",
    }


def test_run_spec_first_target_and_proposal_worker_executes_proposal():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities
            self.calls = []

        def propose_batch(self, inputs):
            self.calls.append(inputs)
            return tuple(
                DraftProposal(
                    sequence_id=row.sequence_id,
                    token_ids=(row.first_target_token,),
                    source_type="fixture",
                )
                for row in inputs
            )

    executor = Executor()
    runner = make_runner()
    runner.rank = 1
    runner.world_size = 4
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        executor,
        capabilities,
    )
    runner.prepare_decode = lambda _seqs: (
        FakeTensor([10, 20], device="cuda:1"),
        FakeTensor([5, 9], device="cuda:1"),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    hidden_rows = ("worker-hidden-8", "worker-hidden-4")
    runner.run_model = lambda *_args, **_kwargs: (
        None,
        hidden_rows,
    )
    selector_calls = []
    original_selector = (
        model_runner.select_tensor_parallel_greedy_tokens
    )

    def worker_selector(
        logits,
        *,
        rank,
        world_size,
        batch_size,
        device,
    ):
        selector_calls.append((
            logits,
            rank,
            world_size,
            batch_size,
            device,
        ))
        return SimpleNamespace(tolist=lambda: [101, 201])

    model_runner.select_tensor_parallel_greedy_tokens = (
        worker_selector
    )
    seqs = _proposal_sequences()
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )
    try:
        rows = runner.run_spec_first_target_and_proposal_batch(
            seqs,
            descriptor,
            (),
        )
    finally:
        model_runner.select_tensor_parallel_greedy_tokens = (
            original_selector
        )

    assert rows is None
    assert len(executor.calls) == 1
    assert tuple(
        row.sequence_id for row in executor.calls[0]
    ) == (8, 4)
    assert tuple(
        row.first_target_token for row in executor.calls[0]
    ) == (101, 201)
    assert executor.calls[0][0].target_hidden == (
        "worker-hidden-8",
    )
    assert executor.calls[0][1].target_hidden == (
        "worker-hidden-4",
    )
    assert selector_calls == [
        (None, 1, 4, 2, "cuda:1"),
    ]


def test_run_spec_first_target_worker_propagates_proposal_failure():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities

        def propose_batch(self, _inputs):
            raise RuntimeError("worker proposal failed")

    runner = make_runner()
    runner.rank = 2
    runner.world_size = 4
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        Executor(),
        capabilities,
    )
    runner.prepare_decode = lambda _seqs: (
        FakeTensor([10, 20], device="cuda:2"),
        FakeTensor([5, 9], device="cuda:2"),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner.run_model = lambda *_args, **_kwargs: (
        None,
        ("worker-hidden-8", "worker-hidden-4"),
    )
    original_selector = (
        model_runner.select_tensor_parallel_greedy_tokens
    )
    model_runner.select_tensor_parallel_greedy_tokens = (
        lambda *_args, **_kwargs:
        SimpleNamespace(tolist=lambda: [101, 201])
    )
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="worker proposal failed",
        ):
            runner.run_spec_first_target_and_proposal_batch(
                _proposal_sequences(),
                descriptor,
                (),
            )
    finally:
        model_runner.select_tensor_parallel_greedy_tokens = (
            original_selector
        )


def test_fused_first_target_prepares_hybrid_state_before_forward():
    capabilities = DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
    )

    class Executor:
        def __init__(self):
            self.capabilities = capabilities

        def propose_batch(self, inputs):
            return tuple(
                DraftProposal(
                    sequence_id=row.sequence_id,
                    token_ids=(),
                    source_type="fixture",
                )
                for row in inputs
            )

    runner = make_runner()
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.register_speculative_proposal_executor(
        "fixture-executor",
        Executor(),
        capabilities,
    )
    seqs = _proposal_sequences()
    for index, seq in enumerate(seqs):
        seq.hybrid_state_slot_id = index
        seq.hybrid_state_generation = index + 1
    calls = []
    runner._prepare_hybrid_state_batch = (
        lambda prepared, released:
        calls.append((
            "prepare_hybrid",
            tuple(prepared),
            tuple(released),
        ))
    )
    runner.kv_offload = None
    runner.prepare_decode = lambda prepared: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = _FakeGreedyLogits([101, 201])

    def run_model(*args, **kwargs):
        calls.append((
            "run_model",
            tuple(
                (
                    lease.request_id,
                    lease.slot_id,
                    lease.generation,
                )
                for lease in runner._last_hybrid_state_leases
            ),
            runner._last_hybrid_state_token_counts,
        ))
        return logits, ("hidden-8", "hidden-4")

    runner.run_model = run_model
    descriptor = SimpleNamespace(
        executor_id="fixture-executor",
        capabilities=capabilities,
    )

    runner.run_spec_first_target_and_proposal_batch(
        seqs,
        descriptor,
        (),
    )

    assert calls == [
        ("prepare_hybrid", seqs, ()),
        (
            "run_model",
            ((8, 0, 1), (4, 1, 2)),
            (1, 1),
        ),
    ]


def test_run_spec_first_target_batch_uses_one_forward_and_orders_rows():
    runner = make_runner()
    calls = []
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=(
            lambda block_id, generation:
            calls.append(("bind", block_id, generation))
        )
    )
    runner.prepare_decode = lambda seqs: (
        calls.append(("prepare_decode", tuple(seqs)))
        or (
            FakeTensor([10, 20]),
            FakeTensor([5, 9]),
        )
    )
    runner._kv_offload_before_forward = (
        lambda: calls.append(("before_forward",))
    )
    runner._kv_offload_after_forward = (
        lambda: calls.append(("after_forward",))
    )
    logits = _FakeGreedyLogits([101, 201])
    hidden_rows = ("hidden-8", "hidden-4")
    runner.run_model = lambda *args, **kwargs: (
        calls.append(("run_model", args, kwargs))
        or (logits, hidden_rows)
    )
    reset_calls = []
    original_reset = model_runner.reset_context
    model_runner.reset_context = lambda: reset_calls.append("reset")
    seqs = _first_target_sequences(0, 0)
    identity_rows = _identity_rows(seqs)
    try:
        rows = runner.run_spec_first_target_batch(
            seqs,
            return_hidden=True,
            return_logits=True,
            kv_block_identity_rows=identity_rows,
        )
    finally:
        model_runner.reset_context = original_reset

    assert tuple(row.sequence_id for row in rows) == (8, 4)
    assert tuple(row.target_token for row in rows) == (101, 201)
    assert tuple(row.target_hidden for row in rows) == hidden_rows
    assert tuple(row.target_logits for row in rows) == (
        ("logits", 0, 101),
        ("logits", 1, 201),
    )
    assert tuple(row.metadata for row in rows) == (
        {"batch_index": 0, "execution_mode": "decode"},
        {"batch_index": 1, "execution_mode": "decode"},
    )
    assert calls[:2] == [
        ("bind", 8, 1),
        ("bind", 4, 2),
    ]
    assert calls[2] == ("prepare_decode", seqs)
    assert calls[3] == ("before_forward",)
    assert calls[4][0] == "run_model"
    assert calls[4][1][2] is False
    assert calls[4][2] == {
        "return_hidden": True,
        "execution_mode": "decode",
    }
    assert calls[5] == ("after_forward",)
    assert logits.argmax_calls == [-1]
    assert logits.index_calls == [0, 1]
    assert reset_calls == ["reset"]


def test_run_offload_spec_first_target_batch_does_not_require_residency_ticket():
    runner = make_runner(kv_offload_mvp0=True)
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=lambda *args: None
    )
    runner.prepare_decode = lambda seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner.run_model = lambda *args, **kwargs: (
        _FakeGreedyLogits([101, 201])
    )
    seqs = _first_target_sequences(0, 0)

    rows = runner.run_spec_first_target_batch(
        seqs,
        kv_block_identity_rows=_identity_rows(seqs),
    )

    assert tuple(row.target_token for row in rows) == (101, 201)


def test_run_spec_first_target_batch_worker_executes_forward_without_result():
    runner = make_runner()
    runner.rank = 1
    runner.prepare_decode = lambda seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = _FakeGreedyLogits([101, 201])
    run_calls = []
    runner.run_model = lambda *args, **kwargs: (
        run_calls.append((args, kwargs))
        or logits
    )

    result = runner.run_spec_first_target_batch(
        _first_target_sequences(0, 0),
    )

    assert result is None
    assert len(run_calls) == 1
    assert logits.argmax_calls == []
    assert logits.index_calls == []


def test_run_spec_first_target_batch_resets_context_after_forward_failure():
    runner = make_runner()
    runner.prepare_decode = lambda seqs: (
        FakeTensor([10, 20]),
        FakeTensor([5, 9]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner.run_model = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("first target forward failed"))
    reset_calls = []
    original_reset = model_runner.reset_context
    model_runner.reset_context = lambda: reset_calls.append("reset")
    try:
        with pytest.raises(
            RuntimeError,
            match="first target forward failed",
        ):
            runner.run_spec_first_target_batch(
                _first_target_sequences(0, 0),
            )
    finally:
        model_runner.reset_context = original_reset

    assert reset_calls == ["reset"]


def test_run_spec_first_target_batch_rejects_non_greedy_before_prepare():
    runner = make_runner()
    prepare_calls = []
    runner.prepare_decode = lambda seqs: prepare_calls.append(seqs)

    with pytest.raises(
        RuntimeError,
        match="greedy temperature",
    ):
        runner.run_spec_first_target_batch(
            _first_target_sequences(0, 0.7),
        )

    assert prepare_calls == []


def test_run_offload_rejects_missing_identity_rows_before_model_step():
    runner = make_runner(kv_offload_mvp0=True)
    runner.kv_offload = SimpleNamespace(
        bind_logical_block_identity=lambda *args: None
    )
    model_step_calls = []
    runner._run_model_step = (
        lambda *args, **kwargs:
        model_step_calls.append((args, kwargs))
    )
    sequence = SimpleNamespace(
        seq_id=7,
        block_table=[1],
    )

    with pytest.raises(ValueError, match="row count mismatch"):
        runner.run(
            [sequence],
            is_prefill=False,
            kv_block_identity_rows=(),
        )

    assert model_step_calls == []


def test_hybrid_state_spec_verify_fails_closed_before_forward():
    runner = make_runner()
    runner.hybrid_state_runtime_bridge = object()

    with pytest.raises(
        RuntimeError,
        match="transactional non-KV state",
    ):
        runner._validate_spec_verify_compatibility(
            seq_count=2,
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
        )


def test_hybrid_state_first_target_is_not_speculative_tail():
    runner = make_runner()
    runner.hybrid_state_runtime_bridge = object()
    seqs = _proposal_sequences()

    runner._validate_spec_first_target_batch(seqs)

    with pytest.raises(
        RuntimeError,
        match="transactional non-KV state",
    ):
        runner._validate_spec_verify_compatibility(
            seq_count=len(seqs),
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
        )


def test_multi_sequence_decode_uses_eager_instead_of_cuda_graph():
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            calls.append(("model", input_ids.values, positions.values))
            return FakeTensor([[1], [2]])

        def compute_logits(self, hidden):
            calls.append(("logits", hidden.values))
            return hidden

    class ForbiddenGraph:
        def replay(self):
            raise AssertionError(
                "multi-sequence decode must not replay a CUDA graph"
            )

    runner = make_runner()
    runner.model = FakeModel()
    runner.graphs = {2: ForbiddenGraph()}
    runner.graph_bs = [2]
    runner.graph_vars = {}

    logits = runner.run_model(
        FakeTensor([10, 20]),
        FakeTensor([53, 54]),
        is_prefill=False,
    )

    assert logits.values == [[1], [2]]
    assert calls == [
        ("model", [10, 20], [53, 54]),
        ("logits", [[1], [2]]),
    ]


def test_single_sequence_decode_uses_eager_when_legacy_graph_state_is_absent():
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            calls.append(("model", input_ids.values, positions.values))
            return FakeTensor([[1]])

        def compute_logits(self, hidden):
            calls.append(("logits", hidden.values))
            return hidden

    runner = make_runner()
    runner.model = FakeModel()

    logits = runner.run_model(
        FakeTensor([10]),
        FakeTensor([53]),
        is_prefill=False,
    )

    assert logits.values == [[1]]
    assert calls == [
        ("model", [10], [53]),
        ("logits", [[1]]),
    ]


def test_single_sequence_decode_still_replays_cuda_graph():
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            raise AssertionError(
                "single-sequence decode should replay the CUDA graph"
            )

        def compute_logits(self, hidden):
            calls.append(("logits", hidden.values))
            return hidden

    class FakeGraph:
        def replay(self):
            calls.append(("replay", None))

    runner = make_runner()
    runner.model = FakeModel()
    runner.graphs = {1: FakeGraph()}
    runner.graph_bs = [1]
    runner.graph_vars = {
        "input_ids": FakeGraphBuffer([0]),
        "positions": FakeGraphBuffer([0]),
        "slot_mapping": FakeGraphBuffer([0]),
        "context_lens": FakeGraphBuffer([0]),
        "block_tables": FakeGraphBuffer([[0]]),
        "outputs": FakeGraphBuffer([[7]]),
    }
    context.set_context(
        False,
        slot_mapping=FakeTensor([4]),
        context_lens=FakeTensor([65]),
        block_tables=FakeTensor([[0]]),
    )

    logits = runner.run_model(
        FakeTensor([10]),
        FakeTensor([64]),
        is_prefill=False,
    )

    assert logits.values == [[7]]
    assert calls == [
        ("replay", None),
        ("logits", [[7]]),
    ]


def test_replay_aware_decode_summary_delegates_to_arena():
    runner = make_runner()
    runner.replay_aware_decode_metadata_arena = SimpleNamespace(
        summary=lambda: {
            "eligible_steps": 3,
            "optimized_steps": 2,
        }
    )

    assert runner.replay_aware_decode_metadata_summary() == {
        "eligible_steps": 3,
        "optimized_steps": 2,
    }


def test_replay_aware_decode_preparation_lands_exact_batch_one():
    calls = []

    class FakeArena:
        def land(self, plan, graph_vars, *, graph_batch_size):
            calls.append(
                (
                    tuple(plan.input_ids),
                    tuple(plan.positions),
                    tuple(plan.block_table_rows),
                    graph_vars,
                    graph_batch_size,
                )
            )
            return SimpleNamespace(
                optimized=True,
                fallback_reason=None,
                input_ids=FakeTensor([12]),
                positions=FakeTensor([2]),
                slot_mapping=FakeTensor([7 * 256 + 2]),
                context_lens=FakeTensor([3]),
                block_tables=FakeTensor([[7]]),
            )

    runner = make_runner(
        replay_aware_decode_metadata=True,
    )
    runner.graph_bs = [1]
    runner.graphs = {1: object()}
    runner.graph_vars = {
        "input_ids": FakeGraphBuffer([0]),
        "positions": FakeGraphBuffer([0]),
        "slot_mapping": FakeGraphBuffer([0]),
        "context_lens": FakeGraphBuffer([0]),
        "block_tables": FakeGraphBuffer([[0]]),
        "outputs": FakeGraphBuffer([[7]]),
    }
    runner.replay_aware_decode_metadata_arena = FakeArena()
    runner._replay_aware_decode_prelanded = False
    sequence = DecodeSequence(
        [10, 11, 12],
        block_table=[7],
    )

    prepared = runner._prepare_replay_aware_decode(
        [sequence]
    )

    assert prepared[0].values == [12]
    assert prepared[1].values == [2]
    assert runner._replay_aware_decode_prelanded is True
    assert calls == [
        (
            (12,),
            (2,),
            ((7,),),
            runner.graph_vars,
            1,
        )
    ]
    active = context.get_context()
    assert active.slot_mapping.values == [7 * 256 + 2]
    assert active.context_lens.values == [3]
    assert active.block_tables.values == [[7]]


def test_replay_aware_decode_preparation_fails_closed_when_disabled():
    runner = make_runner(
        replay_aware_decode_metadata=False,
    )
    runner._replay_aware_decode_prelanded = True
    runner.replay_aware_decode_metadata_arena = SimpleNamespace(
        land=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("disabled path reached arena")
            )
        )
    )

    assert runner._prepare_replay_aware_decode(
        [DecodeSequence([10, 11], block_table=[2])]
    ) is None
    assert runner._replay_aware_decode_prelanded is False


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("batch_two", "active_batch_size_unsupported"),
        ("eager", "enforce_eager"),
        ("kv_offload", "kv_offload_active"),
        ("quest", "quest_active"),
        ("compact_attention", "compact_attention_active"),
        ("kv_quantized", "kv_quantized_eager"),
        ("cpu_offload", "cpu_offload_active"),
        ("kv_cartridge", "kv_cartridge_active"),
        ("graph_state_absent", "legacy_graph_state_absent"),
        ("graph_size_mismatch", "graph_batch_size_mismatch"),
    ],
)
def test_replay_aware_decode_preparation_fails_closed_for_unsupported_paths(
    case,
    expected_reason,
):
    fallback_reasons = []

    class RecordingArena:
        def record_fallback(self, reason):
            fallback_reasons.append(reason)

        def land(self, *_args, **_kwargs):
            raise AssertionError(
                "unsupported path reached metadata landing"
            )

    runner = make_runner(
        replay_aware_decode_metadata=True,
    )
    runner.graph_bs = [1]
    runner.graphs = {1: object()}
    runner.graph_vars = {
        "input_ids": FakeGraphBuffer([0]),
        "positions": FakeGraphBuffer([0]),
        "slot_mapping": FakeGraphBuffer([0]),
        "context_lens": FakeGraphBuffer([0]),
        "block_tables": FakeGraphBuffer([[0]]),
        "outputs": FakeGraphBuffer([[7]]),
    }
    runner.replay_aware_decode_metadata_arena = RecordingArena()
    runner._replay_aware_decode_prelanded = True
    sequences = [
        DecodeSequence([10, 11], block_table=[2])
    ]

    if case == "batch_two":
        sequences.append(
            DecodeSequence([20, 21], block_table=[3])
        )
    elif case == "eager":
        runner.enforce_eager = True
    elif case == "kv_offload":
        runner.config.kv_offload_mvp0 = True
    elif case == "quest":
        runner.config.quest_top_k_blocks = 1
    elif case == "compact_attention":
        runner.config.am_compact_blocks = 1
    elif case == "kv_quantized":
        runner.config.kv_quant_bits = 4
    elif case == "cpu_offload":
        runner.config.cpu_offload = True
    elif case == "kv_cartridge":
        runner.config.kv_cartridge_blocks = 1
    elif case == "graph_state_absent":
        del runner.graph_vars
    elif case == "graph_size_mismatch":
        runner.graph_bs = [2]
        runner.graphs = {2: object()}

    assert runner._prepare_replay_aware_decode(
        sequences
    ) is None
    assert runner._replay_aware_decode_prelanded is False
    assert fallback_reasons == [expected_reason]


def test_prelanded_single_sequence_replay_skips_copy_and_zero():
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            raise AssertionError(
                "prelanded decode should replay the CUDA graph"
            )

        def compute_logits(self, hidden):
            calls.append(("logits", hidden.values))
            return hidden

    class FakeGraph:
        def replay(self):
            calls.append(("replay", None))

    runner = make_runner(
        replay_aware_decode_metadata=True,
    )
    runner.model = FakeModel()
    runner.graphs = {1: FakeGraph()}
    runner.graph_bs = [1]
    runner.graph_vars = {
        "input_ids": FakeGraphBuffer([10]),
        "positions": FakeGraphBuffer([64]),
        "slot_mapping": FakeGraphBuffer([4]),
        "context_lens": FakeGraphBuffer([65]),
        "block_tables": FakeGraphBuffer([[0]]),
        "outputs": FakeGraphBuffer([[7]]),
    }
    runner._replay_aware_decode_prelanded = True
    context.set_context(
        False,
        slot_mapping=FakeTensor([4]),
        context_lens=FakeTensor([65]),
        block_tables=FakeTensor([[0]]),
    )

    logits = runner.run_model(
        FakeTensor([10]),
        FakeTensor([64]),
        is_prefill=False,
    )

    assert logits.values == [[7]]
    assert calls == [
        ("replay", None),
        ("logits", [[7]]),
    ]
    assert all(
        value.zero_calls == 0
        for value in runner.graph_vars.values()
    )
    assert all(
        value.assignments == []
        for value in runner.graph_vars.values()
    )


def test_exact_graph_capacity_reserves_scheduler_invisible_scratch():
    assert model_runner.resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=-1,
        feature_enabled=True,
        scratch_blocks=8,
    ) == (92, 100)
    assert model_runner.resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=80,
        feature_enabled=True,
        scratch_blocks=8,
    ) == (80, 88)
    assert model_runner.resolve_exact_graph_kv_capacity(
        auto_blocks=100,
        requested_visible_blocks=-1,
        feature_enabled=False,
        scratch_blocks=8,
    ) == (100, 100)
    try:
        model_runner.resolve_exact_graph_kv_capacity(
            auto_blocks=100,
            requested_visible_blocks=96,
            feature_enabled=True,
            scratch_blocks=8,
        )
    except ValueError as exc:
        assert "scratch" in str(exc)
    else:
        raise AssertionError("oversized visible plus scratch capacity accepted")


def test_spec_verify_scratch_capacity_covers_worst_row_offset_without_padding():
    required = (
        model_runner.required_spec_verify_capture_scratch_blocks
    )
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(),
        block_size=256,
    ) == 0
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(1, 8),
        block_size=256,
    ) == 8
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(256, 257),
        block_size=256,
    ) == 8
    assert required(
        batch_allowlist=(1, 4),
        query_len_allowlist=(257, 258),
        block_size=256,
    ) == 12


def test_decode_and_spec_verify_scratch_partitions_are_disjoint():
    decode_ids, spec_verify_ids, burst_ids = (
        model_runner.partition_exact_graph_scratch_block_ids(
            visible_blocks=92,
            decode_scratch_blocks=4,
            spec_verify_scratch_blocks=4,
            exact_greedy_burst_scratch_blocks=1,
        )
    )

    assert decode_ids == (92, 93, 94, 95)
    assert spec_verify_ids == (96, 97, 98, 99)
    assert burst_ids == (100,)
    assert set(decode_ids).isdisjoint(spec_verify_ids)
    assert set(decode_ids).isdisjoint(burst_ids)
    assert set(spec_verify_ids).isdisjoint(burst_ids)
    assert min(decode_ids + spec_verify_ids + burst_ids) >= 92


def test_exact_burst_capacity_adds_one_scheduler_invisible_block():
    assert model_runner.resolve_exact_graph_kv_capacity(
        auto_blocks=101,
        requested_visible_blocks=92,
        feature_enabled=True,
        scratch_blocks=9,
    ) == (92, 101)
    decode_ids, spec_verify_ids, burst_ids = (
        model_runner.partition_exact_graph_scratch_block_ids(
            visible_blocks=92,
            decode_scratch_blocks=4,
            spec_verify_scratch_blocks=4,
            exact_greedy_burst_scratch_blocks=1,
        )
    )
    assert max(range(92)) < min(
        decode_ids + spec_verify_ids + burst_ids
    )
    assert burst_ids[0] > max(decode_ids + spec_verify_ids)


def test_exact_burst_scratch_is_reported_by_capacity_snapshot():
    runner = make_runner(exact_greedy_decode_burst=True)
    runner.config.num_kvcache_blocks = 92
    runner._physical_num_kvcache_blocks = 101
    runner._exact_graph_scratch_block_ids = (92, 93, 94, 95)
    runner._spec_verify_capture_scratch_block_ids = (
        96,
        97,
        98,
        99,
    )
    runner._exact_greedy_burst_scratch_block_ids = (100,)

    snapshot = runner.capacity_snapshot()

    assert snapshot["num_kvcache_blocks"] == 92
    assert snapshot["physical_num_kvcache_blocks"] == 101
    assert snapshot["exact_graph_scratch_block_ids"] == [
        92,
        93,
        94,
        95,
    ]
    assert snapshot["spec_verify_capture_scratch_block_ids"] == [
        96,
        97,
        98,
        99,
    ]
    assert snapshot[
        "exact_greedy_burst_scratch_block_ids"
    ] == [100]
    assert 100 not in range(snapshot["num_kvcache_blocks"])


def test_exact_burst_capability_is_fail_closed_and_json_safe():
    runner = make_runner()
    assert runner.exact_greedy_decode_burst_capability() == {
        "available": False,
        "fallback_reason": "disabled",
        "graph_identity_sha256": None,
        "graph_generation": 0,
        "correctness_trace": False,
    }

    runner.config.exact_greedy_decode_burst = True
    assert runner.exact_greedy_decode_burst_capability()[
        "fallback_reason"
    ] == "capture_unavailable"

    class FakeGraph:
        def capability(self):
            return {
                "available": True,
                "graph_identity_sha256": "a" * 64,
                "graph_generation": 7,
                "rank": 0,
                "tensor_parallel_size": 1,
                "block_size": 256,
                "block_table_width": 4,
                "history_capacity": 8,
                "correctness_trace": False,
                "sampled_logit_ordinals": [],
                "quarantine_reason": None,
            }

    runner.exact_greedy_decode_burst_graph = FakeGraph()
    capability = runner.exact_greedy_decode_burst_capability()
    json.dumps(capability, allow_nan=False)
    assert capability == {
        "available": True,
        "fallback_reason": None,
        "graph_identity_sha256": "a" * 64,
        "graph_generation": 7,
        "correctness_trace": False,
    }


def test_model_runner_exact_burst_delegates_once_with_padded_block_table():
    burst_module = sys.modules[
        "tinyvllm.engine.exact_greedy_decode_burst"
    ]
    runner = make_runner(exact_greedy_decode_burst=True)
    calls = []
    materializations = []
    expected = object()

    def prepare_block_tables(rows, name="block_tables"):
        materializations.append((rows, name))
        return FakeTensor([list(row) for row in rows])

    runner.prepare_block_tables_from_rows = prepare_block_tables

    class FakeGraph:
        def capability(self):
            return {
                "available": True,
                "graph_identity_sha256": "b" * 64,
                "graph_generation": 4,
                "rank": 0,
                "tensor_parallel_size": 1,
                "block_size": 256,
                "block_table_width": 4,
                "history_capacity": 8,
                "correctness_trace": False,
                "sampled_logit_ordinals": [],
                "quarantine_reason": None,
            }

        def replay(self, **kwargs):
            assert materializations == []
            block_table = kwargs["block_table_factory"]()
            assert block_table.values == [[5, -1, -1, -1]]
            calls.append(kwargs)
            return expected

    runner.exact_greedy_decode_burst_graph = FakeGraph()
    lease = burst_module.build_exact_greedy_decode_burst_lease(
        sequence_id=7,
        schedule_generation=3,
        graph_generation=4,
        requested_token_count=4,
        authorized_token_count=2,
        initial_completion_count=1,
        initial_sequence_length=2,
        block_table_identity=((5, 9),),
        write_block_id=5,
        write_block_generation=9,
        first_write_position=1,
        last_write_position=2,
        first_physical_slot=1281,
        last_physical_slot=1282,
        remaining_output_tokens=2,
        completion_only=True,
    )
    seq = SimpleNamespace(
        seq_id=7,
        last_token=31,
        block_table=[5],
    )

    steps, original = _capture_profile_steps()
    try:
        result = runner.run_exact_greedy_decode_burst(
            (seq,),
            lease,
        )
    finally:
        model_runner.run_profiled_step = original

    assert result is expected
    assert steps == [{
        "batch_kind": "exact_greedy_decode_burst",
        "is_decode": True,
        "active_sequence_count": 1,
        "request_set_sha256": _profile_request_set_sha256(
            (7,)
        ),
        "dispatch": "cuda_graph",
    }]
    assert len(calls) == 1
    call = calls[0]
    assert call["lease"] is lease
    assert call["initial_token"] == 31
    assert call["block_table"] is None
    assert callable(call["block_table_factory"])
    assert call["continuation_enabled"] is False
    assert materializations == [(
        [[5, -1, -1, -1]],
        "exact_greedy_burst_block_table",
    )]
    assert call["graph_generation"] == 4
    assert call["rank"] == 0
    assert call["tensor_parallel_size"] == 1
    assert call["expected_graph_identity_sha256"] == "b" * 64


def test_model_runner_split_phase_delegates_to_k8_mailbox_backend():
    burst_module = sys.modules[
        "tinyvllm.engine.exact_greedy_decode_burst"
    ]
    runner = make_runner(
        exact_greedy_decode_burst=True,
        exact_greedy_decode_burst_split_phase=True,
        exact_greedy_decode_burst_tokens=8,
    )
    calls = []
    expected = object()
    backend = object()
    runner.exact_greedy_decode_burst_split_phase_backend = backend

    class FakeGraph:
        def capability(self):
            return {
                "available": True,
                "graph_identity_sha256": "b" * 64,
                "graph_generation": 4,
                "rank": 0,
                "tensor_parallel_size": 1,
                "block_size": 256,
                "block_table_width": 4,
                "history_capacity": 8,
                "correctness_trace": False,
                "sampled_logit_ordinals": [],
                "quarantine_reason": None,
            }

        def replay(self, **_kwargs):
            raise AssertionError(
                "split phase used ordinary replay"
            )

        def replay_split_phase(self, **kwargs):
            calls.append(kwargs)
            return expected

    runner.exact_greedy_decode_burst_graph = FakeGraph()
    lease = burst_module.build_exact_greedy_decode_burst_lease(
        sequence_id=7,
        schedule_generation=3,
        graph_generation=4,
        requested_token_count=8,
        authorized_token_count=8,
        initial_completion_count=1,
        initial_sequence_length=249,
        block_table_identity=((5, 9),),
        write_block_id=5,
        write_block_generation=9,
        first_write_position=248,
        last_write_position=255,
        first_physical_slot=5 * 256 + 248,
        last_physical_slot=5 * 256 + 255,
        remaining_output_tokens=8,
        completion_only=True,
    )
    seq = SimpleNamespace(
        seq_id=7,
        last_token=31,
        block_table=[5],
    )

    steps, original = _capture_profile_steps()
    try:
        result = runner.run_exact_greedy_decode_burst(
            (seq,),
            lease,
        )
    finally:
        model_runner.run_profiled_step = original

    assert result is expected
    assert len(calls) == 1
    call = calls[0]
    assert call["lease"] is lease
    assert call["initial_token"] == 31
    assert call["mailbox_backend"] is backend
    assert call["block_table"] is None
    assert callable(call["block_table_factory"])
    assert call["graph_generation"] == 4
    assert call["rank"] == 0
    assert call["tensor_parallel_size"] == 1
    assert call["expected_graph_identity_sha256"] == "b" * 64
    assert steps[0]["dispatch"] == "cuda_graph"


def test_model_runner_split_phase_releases_or_aborts_owned_generation():
    runner = make_runner(
        exact_greedy_decode_burst=True,
        exact_greedy_decode_burst_split_phase=True,
        exact_greedy_decode_burst_tokens=8,
    )
    events = []

    class Backend:
        def release_transaction(self, generation):
            events.append(("release", generation))

        def abort_transaction(self, generation):
            events.append(("abort", generation))

    runner.exact_greedy_decode_burst_split_phase_backend = Backend()
    runner.exact_greedy_decode_burst_split_phase_correctness_backend = (
        Backend()
    )

    runner.release_exact_greedy_decode_burst_split_phase(7)
    runner.abort_exact_greedy_decode_burst_split_phase(
        8,
        correctness_trace=True,
    )

    assert events == [("release", 7), ("abort", 8)]


def test_model_runner_exact_burst_materializes_only_in_lazy_factory():
    tree = ast.parse(open(_MODEL_RUNNER_PATH).read())
    model_runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_exact_greedy_decode_burst"
    )
    materializers = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_block_tables_from_rows"
    ]
    assert len(materializers) == 1
    parent_by_node = {
        child: parent
        for parent in ast.walk(method)
        for child in ast.iter_child_nodes(parent)
    }
    current = materializers[0]
    while current in parent_by_node:
        current = parent_by_node[current]
        if isinstance(current, ast.FunctionDef):
            assert current.name == "materialize_block_table"
            break
    else:
        raise AssertionError(
            "block table materialization is not lazy"
        )
    replay_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replay"
    )
    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in replay_call.keywords
    }
    assert keywords["block_table_factory"] == (
        "materialize_block_table"
    )
    assert keywords["continuation_enabled"] == (
        "self.config.exact_greedy_decode_burst_continuation"
    )


def test_model_runner_invalidates_both_burst_graphs():
    runner = make_runner(exact_greedy_decode_burst=True)
    calls = []

    class FakeGraph:
        def invalidate_continuation(self, reason):
            calls.append(reason)

    runner.exact_greedy_decode_burst_graph = FakeGraph()
    runner.exact_greedy_decode_burst_correctness_graph = FakeGraph()
    runner.invalidate_exact_greedy_decode_burst_continuation(
        "engine_failure:RuntimeError"
    )
    assert calls == [
        "engine_failure:RuntimeError",
        "engine_failure:RuntimeError",
    ]


def test_model_runner_exact_burst_replay_enters_inference_mode():
    tree = ast.parse(open(_MODEL_RUNNER_PATH).read())
    model_runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_exact_greedy_decode_burst"
    )
    decorators = [
        ast.unparse(node) for node in method.decorator_list
    ]
    assert "torch.inference_mode()" in decorators


def test_model_runner_exact_burst_correctness_capture_enters_inference_mode():
    tree = ast.parse(open(_MODEL_RUNNER_PATH).read())
    model_runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "capture_exact_greedy_decode_burst_correctness_graph"
    )
    decorators = [
        ast.unparse(node) for node in method.decorator_list
    ]
    assert "torch.inference_mode()" in decorators


def test_model_runner_exact_burst_rejects_wrong_sequence_before_graph():
    burst_module = sys.modules[
        "tinyvllm.engine.exact_greedy_decode_burst"
    ]
    runner = make_runner(exact_greedy_decode_burst=True)

    class FakeGraph:
        def capability(self):
            return {
                "available": True,
                "graph_identity_sha256": "b" * 64,
                "graph_generation": 4,
                "rank": 0,
                "tensor_parallel_size": 1,
                "block_size": 256,
                "block_table_width": 4,
                "history_capacity": 8,
                "correctness_trace": False,
                "sampled_logit_ordinals": [],
                "quarantine_reason": None,
            }

        def replay(self, **_kwargs):
            raise AssertionError("invalid sequence reached graph replay")

    runner.exact_greedy_decode_burst_graph = FakeGraph()
    lease = burst_module.build_exact_greedy_decode_burst_lease(
        sequence_id=7,
        schedule_generation=3,
        graph_generation=4,
        requested_token_count=4,
        authorized_token_count=2,
        initial_completion_count=1,
        initial_sequence_length=2,
        block_table_identity=((5, 9),),
        write_block_id=5,
        write_block_generation=9,
        first_write_position=1,
        last_write_position=2,
        first_physical_slot=1281,
        last_physical_slot=1282,
        remaining_output_tokens=2,
        completion_only=True,
    )

    fallback = runner.run_exact_greedy_decode_burst(
        (
            SimpleNamespace(
                seq_id=8,
                last_token=31,
                block_table=[5],
            ),
        ),
        lease,
    )
    assert type(fallback).__name__ == (
        "ExactGreedyDecodeBurstFallback"
    )
    assert fallback.fallback_reason == "sequence_identity_drift"
    assert fallback.replay_count == 0


def test_model_runner_exact_burst_capture_owns_static_state_and_pool():
    runner = make_runner(
        exact_greedy_decode_burst=True,
        exact_greedy_decode_burst_split_phase=True,
        exact_greedy_decode_burst_tokens=8,
    )
    runner.config.max_model_len = 512
    runner.config.num_kvcache_blocks = 100
    runner.config.hf_config = SimpleNamespace(vocab_size=32)
    runner._ordinary_graph_generation = 6
    runner._exact_greedy_burst_scratch_block_ids = (100,)
    runner.model = SimpleNamespace(
        compute_logits=lambda hidden: hidden,
    )
    observed = {}
    allocation_devices = []
    marker = object()
    production_backend = object()
    correctness_backend = object()
    split_backends = iter(
        (production_backend, correctness_backend)
    )
    runner._create_exact_greedy_decode_burst_split_phase_backend = (
        lambda: next(split_backends)
    )

    class StaticTensor(FakeCaptureTensor):
        def __init__(self, shape, *, dtype):
            super().__init__(shape)
            self.dtype = dtype
            self.device = "cuda:0"

    class KVView(StaticTensor):
        def data_ptr(self):
            return 1234

        def stride(self):
            return (1, 1, 1)

        def storage_offset(self):
            return 0

    class KVCache:
        device = "cuda:0"

        def __getitem__(self, _index):
            return KVView((2, 1, 100), dtype="float16")

    runner.kv_cache = KVCache()
    original_graph = model_runner.ExactGreedyDecodeBurstGraph
    original_full = getattr(model_runner.torch, "full", None)
    original_zeros = getattr(model_runner.torch, "zeros", None)
    original_cuda = model_runner.torch.cuda

    class FakeBurstGraph:
        @classmethod
        def capture(cls, **kwargs):
            observed.update(kwargs)
            assert kwargs["live_kv_snapshot"]() == (
                1234,
                (2, 1, 100),
                (1, 1, 1),
                0,
            )
            return marker

    model_runner.ExactGreedyDecodeBurstGraph = FakeBurstGraph
    model_runner.torch.full = (
        lambda shape, _value, dtype, device=None: (
            allocation_devices.append(device),
            StaticTensor(
                shape,
                dtype=dtype,
            ),
        )[1]
    )
    model_runner.torch.zeros = (
        lambda shape, dtype, device=None: (
            allocation_devices.append(device),
            StaticTensor(
                shape,
                dtype=dtype,
            ),
        )[1]
    )
    model_runner.torch.cuda = SimpleNamespace(
        graph_pool_handle=lambda: "private-pool",
        CUDAGraph=lambda: object(),
        graph=lambda graph, pool=None: (graph, pool),
        synchronize=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    try:
        result = runner._capture_exact_greedy_decode_burst()
        production_observed = dict(observed)
        production_allocation_devices = list(
            allocation_devices
        )
        observed.clear()
        allocation_devices.clear()
        correctness_result = (
            runner._capture_exact_greedy_decode_burst(
                correctness_trace=True,
                sampled_logit_ordinals=(0, 63, 126),
            )
        )
        correctness_observed = dict(observed)
        correctness_allocation_devices = list(
            allocation_devices
        )
    finally:
        model_runner.ExactGreedyDecodeBurstGraph = original_graph
        if original_full is None:
            delattr(model_runner.torch, "full")
        else:
            model_runner.torch.full = original_full
        if original_zeros is None:
            delattr(model_runner.torch, "zeros")
        else:
            model_runner.torch.zeros = original_zeros
        model_runner.torch.cuda = original_cuda

    assert result is marker
    assert runner.exact_greedy_decode_burst_graph is marker
    assert (
        runner.exact_greedy_decode_burst_split_phase_backend
        is production_backend
    )
    assert production_observed["graph_pool"] == "private-pool"
    assert production_observed["graph_generation"] == 6
    assert production_observed["scratch_block_id"] == 100
    assert production_observed["block_size"] == 256
    assert production_observed["correctness_trace"] is False
    assert production_observed["sampled_logit_ordinals"] == ()
    assert production_observed["tensors"]["input_token"].shape == (1,)
    assert production_observed["tensors"]["block_table"].shape == (1, 2)
    assert production_observed["tensors"]["token_history"].shape == (
        256,
    )
    assert production_allocation_devices == ["cuda:0"] * 7
    assert correctness_observed["correctness_trace"] is True
    assert correctness_observed["sampled_logit_ordinals"] == (
        0,
        63,
        126,
    )
    assert correctness_observed["tensors"]["sampled_logits"].shape == (
        3,
        32,
    )
    assert correctness_observed["tensors"]["sample_ordinals"].shape == (
        3,
    )
    assert correctness_allocation_devices == ["cuda:0"] * 9
    assert correctness_result is marker
    assert (
        runner.exact_greedy_decode_burst_correctness_graph
        is marker
    )
    assert (
        runner
        .exact_greedy_decode_burst_split_phase_correctness_backend
        is correctness_backend
    )


def test_scratch_blocks_are_above_scheduler_visible_range():
    runner = make_runner()
    runner.config.num_kvcache_blocks = 92
    runner._physical_num_kvcache_blocks = 100
    runner._exact_graph_scratch_block_ids = tuple(range(92, 100))
    assert runner._exact_graph_scratch_block_ids == tuple(range(92, 100))
    assert max(range(runner.config.num_kvcache_blocks)) < min(
        runner._exact_graph_scratch_block_ids
    )
    assert runner._exact_graph_scratch_slots(batch_size=8) == tuple(
        block * runner.block_size for block in range(92, 100)
    )


def _make_capture_runner(*, feature_enabled):
    runner = make_runner(
        multi_sequence_cuda_graphs=feature_enabled,
    )
    runner.config.hf_config = SimpleNamespace(hidden_size=16)
    runner.config.max_num_seqs = 8
    runner.config.max_model_len = 512

    class FakeModel:
        def __call__(self, input_ids, positions):
            del input_ids, positions
            return FakeCaptureTensor((8, 16))

        def compute_logits(self, hidden):
            return hidden

    class FakeGraph:
        def replay(self):
            pass

        def pool(self):
            return "pool"

    class FakeGraphContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    runner.model = FakeModel()
    model_runner.torch.zeros = (
        lambda *shape, **kwargs: FakeCaptureTensor(
            shape[0] if len(shape) > 1 else shape[0]
        )
    )
    model_runner.torch.cuda.CUDAGraph = FakeGraph
    model_runner.torch.cuda.graph = (
        lambda graph, pool=None: FakeGraphContext()
    )
    model_runner.torch.cuda.synchronize = lambda: None
    model_runner.torch.cuda.memory_allocated = lambda: 0
    model_runner.torch.cuda.memory_reserved = lambda: 0
    return runner


def test_feature_enabled_startup_captures_only_batch_one():
    runner = _make_capture_runner(feature_enabled=True)
    runner.capture_cudagraph()
    assert tuple(runner.graph_bs) == (1,)
    assert set(runner.graphs) == {1}


def test_feature_disabled_startup_inventory_is_unchanged():
    runner = _make_capture_runner(feature_enabled=False)
    runner.capture_cudagraph()
    assert runner.graph_bs[:4] == [1, 2, 4, 8]


def test_graph_tail_capture_binds_batch_one_static_output():
    runner = _make_capture_runner(feature_enabled=False)
    runner.config.graph_resident_greedy_tail = True
    observed = {}
    marker = object()
    original = getattr(
        model_runner,
        "GraphResidentGreedyTail",
        None,
    )

    class FakeTail:
        @classmethod
        def capture(cls, **kwargs):
            observed.update(kwargs)
            return marker

    model_runner.GraphResidentGreedyTail = FakeTail
    try:
        runner.capture_cudagraph()
    finally:
        if original is None:
            delattr(model_runner, "GraphResidentGreedyTail")
        else:
            model_runner.GraphResidentGreedyTail = original

    assert runner._ordinary_graph_generation == 1
    assert runner.graph_resident_greedy_tail is marker
    assert observed["static_hidden"] is runner.graph_vars["outputs"]
    assert observed["compute_logits"] == runner.model.compute_logits
    assert observed["float32_dtype"] == model_runner.torch.float32
    assert observed["graph_generation"] == 1
    assert observed["rank"] == 0
    assert observed["stats"] is runner.graph_resident_greedy_tail_stats


def test_capture_cudagraph_initializes_exact_burst_after_generation():
    runner = _make_capture_runner(feature_enabled=False)
    runner.config.exact_greedy_decode_burst = True
    observed = []
    runner._capture_exact_greedy_decode_burst = (
        lambda: observed.append(
            runner._ordinary_graph_generation
        )
    )

    runner.capture_cudagraph()

    assert observed == [1]


def _graph_tail_decode_runner(*, temperature=0.0):
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            raise AssertionError(
                "ordinary decode should replay the transformer graph"
            )

        def compute_logits(self, hidden):
            calls.append(("external_logits", hidden.values))
            return hidden

    class FakeGraph:
        def replay(self):
            calls.append(("transformer_replay", None))

    class FakeTail:
        def __init__(self):
            self.replay_calls = 0

        def matches(self, **kwargs):
            calls.append(
                (
                    "tail_matches",
                    kwargs["graph_generation"],
                    kwargs["rank"],
                )
            )
            return True

        def replay(self, **kwargs):
            self.replay_calls += 1
            calls.append(
                (
                    "tail_replay",
                    kwargs["graph_generation"],
                    kwargs["rank"],
                )
            )
            return GraphResidentGreedyTailReplay(
                logits=_TraceTensor([[0.0, 9.0, 1.0]]),
                token_ids=SimpleNamespace(tolist=lambda: [1]),
            )

    runner = make_runner(
        zero_temperature_greedy_fast_path=True,
        graph_resident_greedy_tail=True,
    )
    runner.model = FakeModel()
    runner.graphs = {1: FakeGraph()}
    runner.graph_bs = [1]
    runner.graph_vars = {
        "input_ids": FakeGraphBuffer([0]),
        "positions": FakeGraphBuffer([0]),
        "slot_mapping": FakeGraphBuffer([0]),
        "context_lens": FakeGraphBuffer([0]),
        "block_tables": FakeGraphBuffer([[0]]),
        "outputs": FakeGraphBuffer([[7]]),
    }
    runner._ordinary_graph_generation = 3
    runner.graph_resident_greedy_tail = FakeTail()
    context.set_context(
        False,
        slot_mapping=FakeTensor([4]),
        context_lens=FakeTensor([65]),
        block_tables=FakeTensor([[0]]),
    )
    return runner, calls, temperature


def test_graph_tail_decides_before_transformer_replay_and_skips_logits():
    runner, calls, temperature = _graph_tail_decode_runner()

    result = runner.run_model(
        FakeTensor([10]),
        FakeTensor([64]),
        is_prefill=False,
        graph_tail_temperatures=(temperature,),
        graph_tail_do_sample=True,
        graph_tail_batch_kind=None,
    )

    assert isinstance(result, GraphResidentGreedyTailReplay)
    assert calls == [
        ("tail_matches", 3, 0),
        ("transformer_replay", None),
        ("tail_replay", 3, 0),
    ]


def test_graph_tail_ineligible_step_preserves_external_logits_path():
    runner, calls, _temperature = _graph_tail_decode_runner(
        temperature=0.7,
    )

    result = runner.run_model(
        FakeTensor([10]),
        FakeTensor([64]),
        is_prefill=False,
        graph_tail_temperatures=(0.7,),
        graph_tail_do_sample=True,
        graph_tail_batch_kind=None,
    )

    assert result.values == [[7]]
    assert calls == [
        ("tail_matches", 3, 0),
        ("transformer_replay", None),
        ("external_logits", [[7]]),
    ]
    assert runner.graph_resident_greedy_tail_stats.summary()[
        "fallback_counts"
    ] == {"nonzero_temperature": 1}


def test_graph_tail_replay_failure_never_falls_back_or_replays_twice():
    runner, calls, temperature = _graph_tail_decode_runner()

    def fail_replay(**kwargs):
        calls.append(
            (
                "tail_replay_failure",
                kwargs["graph_generation"],
                kwargs["rank"],
            )
        )
        raise RuntimeError("tail replay failed")

    runner.graph_resident_greedy_tail.replay = fail_replay
    try:
        runner.run_model(
            FakeTensor([10]),
            FakeTensor([64]),
            is_prefill=False,
            graph_tail_temperatures=(temperature,),
            graph_tail_do_sample=True,
            graph_tail_batch_kind=None,
        )
    except RuntimeError as error:
        assert str(error) == "tail replay failed"
    else:
        raise AssertionError("tail replay failure did not propagate")

    assert calls == [
        ("tail_matches", 3, 0),
        ("transformer_replay", None),
        ("tail_replay_failure", 3, 0),
    ]


def test_exact_graph_identity_and_static_byte_estimate_are_exact():
    runner = make_runner()
    runner.config.hf_config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=1024,
        torch_dtype=SimpleNamespace(itemsize=2),
    )
    runner.kv_cache = FakeTensor([], device="cuda:0")
    context.set_context(
        False,
        slot_mapping=FakeTensor([0, 256, 512, 768]),
        context_lens=FakeTensor([1, 1, 1, 1]),
        block_tables=FakeTensor([[0, 1]] * 4),
    )
    identity = runner._build_multi_sequence_graph_identity(
        FakeTensor([10, 20, 30, 40]),
        context.get_context(),
    )
    assert identity.graph_batch_size == identity.active_batch_size == 4
    assert identity.page_table_width == 2
    assert identity.flash_attn_version == "2.6.3"
    assert identity.multi_processor_count == 108
    assert runner._estimate_exact_graph_static_bytes(
        batch_size=4,
        page_table_width=2,
    ) > 0
    wider_context = SimpleNamespace(
        block_tables=FakeTensor([[0, 1, 2]] * 4),
    )
    wider = runner._build_multi_sequence_graph_identity(
        FakeTensor([10, 20, 30, 40]),
        wider_context,
    )
    assert wider.sha256 != identity.sha256

    original_properties = (
        model_runner.torch.cuda.get_device_properties
    )
    model_runner.torch.cuda.get_device_properties = (
        lambda device: SimpleNamespace(multi_processor_count=120)
    )
    try:
        different_sm = runner._build_multi_sequence_graph_identity(
            FakeTensor([10, 20, 30, 40]),
            context.get_context(),
        )
    finally:
        model_runner.torch.cuda.get_device_properties = (
            original_properties
        )
    assert different_sm.sha256 != identity.sha256

    runner.config.hf_config.num_attention_heads = 32
    different_heads = runner._build_multi_sequence_graph_identity(
        FakeTensor([10, 20, 30, 40]),
        context.get_context(),
    )
    assert different_heads.sha256 != identity.sha256

    model_runner.flash_attn.__version__ = "2.6.2"
    try:
        runner._build_multi_sequence_graph_identity(
            FakeTensor([10, 20, 30, 40]),
            context.get_context(),
        )
    except ValueError as exc:
        assert "2.6.3" in str(exc)
    else:
        raise AssertionError("unsupported FlashAttention version accepted")
    finally:
        model_runner.flash_attn.__version__ = "2.6.3"


def _make_exact_dispatch_runner():
    cache_module = sys.modules[
        "tinyvllm.engine.exact_cuda_graph_cache"
    ]
    runner = make_runner(
        multi_sequence_cuda_graphs=True,
    )
    runner.config.hf_config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=16,
        torch_dtype=SimpleNamespace(itemsize=2),
    )
    runner.kv_cache = FakeTensor([], device="cuda:0")
    runner.exact_cuda_graph_cache = cache_module.ExactCudaGraphCache(
        cache_module.ExactCudaGraphCacheConfig(
            enabled=True,
            batch_allowlist=(2, 4, 8),
            min_observations=3,
            max_entries=8,
            max_static_bytes=64 * 1024 * 1024,
            max_reserved_bytes=512 * 1024 * 1024,
            max_total_capture_ns=5_000_000_000,
            max_single_capture_ns=2_000_000_000,
        )
    )
    runner.last_cuda_graph_dispatch_event = None
    runner._cuda_graph_step_id = 0
    runner._cuda_graph_request_ids_hash = "request-hash"

    class FakeModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            del positions, input_embeds
            return FakeTensor(
                [[value] for value in input_ids.values]
            )

        def compute_logits(self, hidden):
            return hidden

    runner.model = FakeModel()

    class FakeGraph:
        def replay(self):
            pass

    def capture(
        *,
        identity,
        input_ids,
        positions,
        context,
    ):
        del input_ids, positions, context
        return cache_module.ExactCudaGraphEntry(
            identity=identity,
            identity_sha256=identity.sha256,
            graph=FakeGraph(),
            tensors={
                "input_ids": FakeGraphBuffer(),
                "positions": FakeGraphBuffer(),
                "slot_mapping": FakeGraphBuffer(),
                "context_lens": FakeGraphBuffer(),
                "block_tables": FakeGraphBuffer(),
                "outputs": FakeTensor([[101], [102], [103], [104]]),
            },
            static_bytes=4096,
            capture_duration_ns=100,
            allocated_delta_bytes=0,
            reserved_delta_bytes=1024,
        )

    runner._capture_exact_multi_sequence_graph = capture
    return runner


def _run_exact_decode(runner):
    context.set_context(
        False,
        slot_mapping=FakeTensor([0, 256, 512, 768]),
        context_lens=FakeTensor([1, 1, 1, 1]),
        block_tables=FakeTensor([[0, 1]] * 4),
    )
    logits = runner.run_model(
        FakeTensor([10, 20, 30, 40]),
        FakeTensor([1, 1, 1, 1]),
        is_prefill=False,
    )
    return logits, runner.cuda_graph_dispatch_observation()


def test_three_successful_eager_steps_capture_post_step_and_fourth_replays():
    runner = _make_exact_dispatch_runner()
    results = [_run_exact_decode(runner) for _ in range(4)]
    events = [event for _, event in results]
    assert [event["dispatch"] for event in events] == [
        "eager",
        "eager",
        "eager",
        "graph",
    ]
    assert [event["capture_attempted"] for event in events] == [
        False,
        False,
        True,
        False,
    ]
    assert results[2][0].values == [[10], [20], [30], [40]]
    assert results[3][0].values == [[101], [102], [103], [104]]
    assert events[2]["fallback_reason"] == "cold_identity"
    assert events[3]["cache_state"] == "ready"
    assert events[3]["graph_identity_sha256"]


def test_exact_ready_entry_never_rounds_batch_or_page_table_width():
    runner = _make_exact_dispatch_runner()
    for _ in range(3):
        _run_exact_decode(runner)
    _, ready_event = _run_exact_decode(runner)
    ready_sha = ready_event["graph_identity_sha256"]

    cases = (
        (
            FakeTensor([10, 20, 30]),
            FakeTensor([[0, 1]] * 3),
        ),
        (
            FakeTensor([10, 20, 30, 40]),
            FakeTensor([[0]] * 4),
        ),
        (
            FakeTensor([10, 20, 30, 40]),
            FakeTensor([[0, 1, 2]] * 4),
        ),
        (
            FakeTensor([10, 20, 30, 40, 50]),
            FakeTensor([[0, 1]] * 5),
        ),
    )
    for input_ids, block_tables in cases:
        batch_size = input_ids.size(0)
        context.set_context(
            False,
            slot_mapping=FakeTensor(
                [index * 256 for index in range(batch_size)]
            ),
            context_lens=FakeTensor([1] * batch_size),
            block_tables=block_tables,
        )
        logits = runner.run_model(
            input_ids,
            FakeTensor([1] * batch_size),
            is_prefill=False,
        )
        event = runner.cuda_graph_dispatch_observation()
        assert event["dispatch"] == "eager"
        assert event["graph_identity_sha256"] != ready_sha
        assert logits.values == [
            [value] for value in input_ids.values
        ]


def test_multi_sequence_graph_guards_report_exact_fallback_reasons():
    cases = (
        (
            {"multi_sequence_cuda_graphs": False},
            {},
            {},
            "decode",
            False,
            None,
            False,
            "feature_disabled",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {"enforce_eager": True},
            {},
            "decode",
            False,
            None,
            False,
            "enforce_eager",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {},
            "prefill",
            True,
            None,
            False,
            "unsupported_mode",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {},
            "spec_verify",
            False,
            None,
            False,
            "feature_disabled",
        ),
        (
            {
                "multi_sequence_cuda_graphs": True,
                "kv_quant_bits": 4,
            },
            {},
            {},
            "decode",
            False,
            None,
            False,
            "incompatible_feature",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {"quest_top_k_blocks": 1},
            "decode",
            False,
            None,
            False,
            "incompatible_feature",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {"am_compact_blocks": 1},
            "decode",
            False,
            None,
            False,
            "incompatible_feature",
        ),
        (
            {
                "multi_sequence_cuda_graphs": True,
                "cpu_offload": True,
            },
            {},
            {},
            "decode",
            False,
            None,
            False,
            "incompatible_feature",
        ),
        (
            {
                "multi_sequence_cuda_graphs": True,
                "kv_offload_mvp0": True,
            },
            {},
            {},
            "decode",
            False,
            None,
            False,
            "incompatible_feature",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {},
            "decode",
            False,
            FakeTensor([[]]),
            False,
            "incompatible_feature",
        ),
        (
            {"multi_sequence_cuda_graphs": True},
            {},
            {},
            "decode",
            False,
            None,
            True,
            "incompatible_feature",
        ),
    )
    for (
        config_overrides,
        runner_overrides,
        context_overrides,
        mode,
        is_prefill,
        input_embeds,
        return_hidden,
        expected_reason,
    ) in cases:
        runner = _make_exact_dispatch_runner()
        for name, value in config_overrides.items():
            setattr(runner.config, name, value)
        for name, value in runner_overrides.items():
            setattr(runner, name, value)
            if name == "enforce_eager":
                runner.enforce_eager = value
        context.set_context(
            mode=mode,
            slot_mapping=FakeTensor([0, 256, 512, 768]),
            context_lens=FakeTensor([1, 1, 1, 1]),
            block_tables=FakeTensor([[0, 1]] * 4),
            **context_overrides,
        )
        runner.run_model(
            FakeTensor([10, 20, 30, 40]),
            FakeTensor([1, 1, 1, 1]),
            is_prefill=is_prefill,
            input_embeds=input_embeds,
            return_hidden=return_hidden,
            execution_mode=mode,
        )
        event = (
            runner.spec_verify_graph_dispatch_observation()
            if mode == "spec_verify"
            else runner.cuda_graph_dispatch_observation()
        )
        assert event["dispatch"] == "eager"
        assert event["fallback_reason"] == expected_reason


def test_non_allowlisted_and_invalid_identity_fail_closed():
    runner = _make_exact_dispatch_runner()
    context.set_context(
        False,
        slot_mapping=FakeTensor([0, 256, 512]),
        context_lens=FakeTensor([1, 1, 1]),
        block_tables=FakeTensor([[0, 1]] * 3),
    )
    runner.run_model(
        FakeTensor([10, 20, 30]),
        FakeTensor([1, 1, 1]),
        is_prefill=False,
    )
    event = runner.cuda_graph_dispatch_observation()
    assert event["dispatch"] == "eager"
    assert event["fallback_reason"] == "batch_not_allowlisted"

    runner = _make_exact_dispatch_runner()
    runner._build_multi_sequence_graph_identity = (
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(ValueError("invalid identity"))
    )
    _run_exact_decode(runner)
    event = runner.cuda_graph_dispatch_observation()
    assert event["dispatch"] == "eager"
    assert event["fallback_reason"] == "identity_invalid"


def test_dispatch_event_schema_is_complete_and_ordered():
    os.environ["TINYVLLM_SOURCE_SHA256"] = "source-sha"
    try:
        runner = _make_exact_dispatch_runner()
        _run_exact_decode(runner)
        event = runner.cuda_graph_dispatch_observation()
        assert tuple(event) == model_runner.DISPATCH_EVENT_FIELDS
        assert event["source_sha256"] == "source-sha"
        assert event["fallback_reason"] in (
            sys.modules[
                "tinyvllm.engine.exact_cuda_graph_cache"
            ].FALLBACK_REASONS
        )
        assert event["dispatch"] in {"eager", "graph"}
        assert event["step_id"] == 1
    finally:
        os.environ.pop("TINYVLLM_SOURCE_SHA256", None)


def test_every_fallback_reason_and_graph_hit_use_closed_event_schema():
    cache_module = sys.modules[
        "tinyvllm.engine.exact_cuda_graph_cache"
    ]
    runner = _make_exact_dispatch_runner()
    os.environ["TINYVLLM_SOURCE_SHA256"] = "source-sha"
    try:
        for reason in cache_module.FALLBACK_REASONS:
            runner._publish_cuda_graph_dispatch_event(
                mode="decode",
                active_batch_size=4,
                page_table_width=2,
                effective_num_splits=2,
                graph_identity_sha256="identity-sha",
                dispatch="eager",
                cache_state="rejected",
                observation_count=3,
                fallback_reason=reason,
                capture_attempted=False,
            )
            event = runner.cuda_graph_dispatch_observation()
            assert tuple(event) == model_runner.DISPATCH_EVENT_FIELDS
            assert event["fallback_reason"] == reason
            assert event["source_sha256"] == "source-sha"
        runner._publish_cuda_graph_dispatch_event(
            mode="decode",
            active_batch_size=4,
            page_table_width=2,
            effective_num_splits=2,
            graph_identity_sha256="identity-sha",
            dispatch="graph",
            cache_state="ready",
            observation_count=3,
            fallback_reason=None,
            capture_attempted=False,
        )
        event = runner.cuda_graph_dispatch_observation()
        assert tuple(event) == model_runner.DISPATCH_EVENT_FIELDS
        assert event["dispatch"] == "graph"
        assert event["fallback_reason"] is None
        assert event["step_id"] == len(cache_module.FALLBACK_REASONS) + 1
    finally:
        os.environ.pop("TINYVLLM_SOURCE_SHA256", None)


def test_run_hashes_canonical_sorted_sequence_ids_before_dispatch():
    runner = make_runner()
    runner.prepare_decode = (
        lambda seqs: (FakeTensor([1, 2, 3]), FakeTensor([0, 0, 0]))
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    observed = {}

    def run_model(input_ids, positions, is_prefill):
        del input_ids, positions, is_prefill
        observed["request_ids_hash"] = (
            runner._cuda_graph_request_ids_hash
        )
        return FakeTensor([[1], [2], [3]])

    runner.run_model = run_model
    seqs = [
        make_sequence(9),
        make_sequence(2),
        make_sequence(5),
    ]
    runner.run(seqs, is_prefill=False, do_sample=False)
    expected = hashlib.sha256(
        json.dumps(
            [2, 5, 9],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert observed["request_ids_hash"] == expected


def _make_step_logits_run_runner(*, rank=0):
    runner = make_runner()
    runner.rank = rank
    runner._record_step_logits = True
    runner._last_step_logits_cpu = FakeIndexedTensor("stale")
    runner.prepare_decode = (
        lambda seqs: (FakeTensor([1, 2]), FakeTensor([0, 0]))
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner.prepare_sample = lambda seqs: FakeTensor([1.0] * len(seqs))
    return runner


def test_run_records_selected_rank_zero_sampling_logits_before_sampler():
    runner = _make_step_logits_run_runner()
    logits = FakeIndexedTensor([[1.0, 2.0], [3.0, 4.0]])
    runner.run_model = lambda *args: logits
    sampled = {}

    class SampledTokens:
        def tolist(self):
            return [7, 8]

    def sampler(selected_logits, temperatures):
        sampled["logits"] = selected_logits
        sampled["temperatures"] = temperatures
        return SampledTokens()

    runner.sampler = sampler
    seqs = [make_sequence(1), make_sequence(2)]

    assert runner.run(seqs, is_prefill=False) == [7, 8]
    assert sampled["logits"] is logits
    recorded = runner.last_step_logits()
    assert recorded is not None
    assert recorded.values == ("cloned", logits.values)
    assert logits.trace[:3] == [
        ("detach", None),
        ("float", None),
        ("cpu", None),
    ]


def _load_real_config_class():
    module_name = "tinyvllm_config_greedy_fast_path_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original_transformers = sys.modules.get("transformers")
    try:
        sys.modules["transformers"] = fake_transformers
        return _load_source_module(module_name, _CONFIG_PATH).Config
    finally:
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers
        sys.modules.pop(module_name, None)


def test_zero_temperature_greedy_fast_path_config_is_fail_closed():
    Config = _load_real_config_class()
    assert (
        Config.__dataclass_fields__[
            "zero_temperature_greedy_fast_path"
        ].default
        is False
    )
    with tempfile.TemporaryDirectory() as model:
        try:
            Config(
                model=model,
                zero_temperature_greedy_fast_path=1,
            )
        except ValueError as error:
            assert str(error) == (
                "zero_temperature_greedy_fast_path must be a bool"
            )
        else:
            raise AssertionError(
                "non-boolean greedy fast-path control was accepted"
            )


def test_graph_resident_greedy_tail_config_is_fail_closed():
    Config = _load_real_config_class()
    assert (
        Config.__dataclass_fields__[
            "graph_resident_greedy_tail"
        ].default
        is False
    )
    with tempfile.TemporaryDirectory() as model:
        try:
            Config(
                model=model,
                graph_resident_greedy_tail=1,
            )
        except ValueError as error:
            assert str(error) == (
                "graph_resident_greedy_tail must be a bool"
            )
        else:
            raise AssertionError(
                "non-boolean graph-tail control was accepted"
            )


def test_exact_greedy_decode_burst_config_is_strict_and_default_off():
    Config = _load_real_config_class()
    fields = Config.__dataclass_fields__
    assert fields["exact_greedy_decode_burst"].default is False
    assert (
        fields["exact_greedy_decode_burst_continuation"].default
        is False
    )
    assert (
        fields["exact_greedy_decode_burst_split_phase"].default
        is False
    )
    assert (
        fields[
            "exact_greedy_decode_burst_ragged_coalescing"
        ].default
        is False
    )
    assert fields["exact_greedy_decode_burst_tokens"].default == 4
    with tempfile.TemporaryDirectory() as model:
        for invalid in (0, 1, None, "true"):
            with pytest.raises(
                ValueError,
                match=(
                    "^exact_greedy_decode_burst must be a bool$"
                ),
            ):
                Config(
                    model=model,
                    exact_greedy_decode_burst=invalid,
                )
        for invalid in (0, 1, None, "true"):
            with pytest.raises(
                ValueError,
                match=(
                    "^exact_greedy_decode_burst_continuation "
                    "must be a bool$"
                ),
            ):
                Config(
                    model=model,
                    exact_greedy_decode_burst_continuation=invalid,
                )
        for invalid in (0, 1, None, "true"):
            with pytest.raises(
                ValueError,
                match=(
                    "^exact_greedy_decode_burst_split_phase "
                    "must be a bool$"
                ),
            ):
                Config(
                    model=model,
                    exact_greedy_decode_burst_split_phase=invalid,
                )
        for invalid in (0, 1, None, "true"):
            with pytest.raises(
                ValueError,
                match=(
                    "^exact_greedy_decode_burst_ragged_"
                    "coalescing must be a bool$"
                ),
            ):
                Config(
                    model=model,
                    exact_greedy_decode_burst_ragged_coalescing=(
                        invalid
                    ),
                )
        for invalid in (False, True, 1, 9, 4.0, None):
            with pytest.raises(
                ValueError,
                match=(
                    "^exact_greedy_decode_burst_tokens must "
                    "be an integer in \\[2, 8\\]$"
                ),
            ):
                Config(
                    model=model,
                    exact_greedy_decode_burst_tokens=invalid,
                )
        with pytest.raises(
            ValueError,
            match=(
                "^split phase requires exact_greedy_decode_burst$"
            ),
        ):
            Config(
                model=model,
                exact_greedy_decode_burst_split_phase=True,
                exact_greedy_decode_burst_tokens=8,
            )
        with pytest.raises(
            ValueError,
            match="^split phase requires K8$",
        ):
            Config(
                model=model,
                exact_greedy_decode_burst=True,
                exact_greedy_decode_burst_split_phase=True,
                exact_greedy_decode_burst_tokens=4,
            )
        with pytest.raises(
            ValueError,
            match=(
                "^split phase cannot compose with continuation$"
            ),
        ):
            Config(
                model=model,
                exact_greedy_decode_burst=True,
                exact_greedy_decode_burst_continuation=True,
                exact_greedy_decode_burst_split_phase=True,
                exact_greedy_decode_burst_tokens=8,
            )
        with pytest.raises(
            ValueError,
            match=(
                "^ragged coalescing requires "
                "exact_greedy_decode_burst$"
            ),
        ):
            Config(
                model=model,
                exact_greedy_decode_burst_ragged_coalescing=True,
            )
        with pytest.raises(
            ValueError,
            match="^ragged coalescing requires split phase$",
        ):
            Config(
                model=model,
                exact_greedy_decode_burst=True,
                exact_greedy_decode_burst_tokens=8,
                exact_greedy_decode_burst_ragged_coalescing=True,
            )
        with pytest.raises(
            ValueError,
            match="^ragged coalescing requires K8$",
        ):
            Config(
                model=model,
                exact_greedy_decode_burst=True,
                exact_greedy_decode_burst_split_phase=True,
                exact_greedy_decode_burst_tokens=4,
                exact_greedy_decode_burst_ragged_coalescing=True,
            )
        enabled = Config(
            model=model,
            exact_greedy_decode_burst=True,
            exact_greedy_decode_burst_split_phase=True,
            exact_greedy_decode_burst_tokens=8,
            exact_greedy_decode_burst_ragged_coalescing=True,
        )
    assert enabled.exact_greedy_decode_burst is True
    assert enabled.exact_greedy_decode_burst_continuation is False
    assert enabled.exact_greedy_decode_burst_split_phase is True
    assert (
        enabled.exact_greedy_decode_burst_ragged_coalescing
        is True
    )
    assert enabled.exact_greedy_decode_burst_tokens == 8


class _GreedyFastPathLogits:
    def __init__(self, values, *, shape=None):
        self.values = tuple(tuple(row) for row in values)
        self.shape = (
            tuple(shape)
            if shape is not None
            else (len(self.values), len(self.values[0]))
        )
        self.trace = []

    def to(self, dtype):
        self.trace.append(("to", dtype))
        return self

    def argmax(self, dim):
        self.trace.append(("argmax", dim))
        tokens = [
            max(range(len(row)), key=row.__getitem__)
            for row in self.values
        ]
        return SimpleNamespace(
            tolist=lambda: self.trace.append(("tolist", None))
            or tokens
        )


def test_greedy_fast_path_uses_exact_float32_argmax():
    runner = make_runner(
        zero_temperature_greedy_fast_path=True,
    )
    prepare_calls = []
    sampler_calls = []
    runner.prepare_sample = (
        lambda seqs: prepare_calls.append(tuple(seqs))
    )
    runner.sampler = (
        lambda logits, temperatures: sampler_calls.append(
            (logits, temperatures)
        )
    )
    logits = _GreedyFastPathLogits([[1.0, 4.0, 2.0]])
    original_values = logits.values

    token_ids = (
        runner._sample_tokens_with_optional_greedy_fast_path(
            logits,
            [SimpleNamespace(temperature=0.0)],
            batch_kind=None,
        )
    )

    assert token_ids == [1]
    assert logits.values == original_values
    assert logits.trace == [
        ("to", model_runner.torch.float32),
        ("argmax", -1),
        ("tolist", None),
    ]
    assert prepare_calls == []
    assert sampler_calls == []
    assert runner.zero_temperature_greedy_fast_path_summary() == {
        "eligible_steps": 1,
        "optimized_steps": 1,
        "avoided_temperature_h2d_bytes": 4,
        "avoided_softmax_calls": 1,
        "avoided_gumbel_rng_calls": 1,
        "avoided_stochastic_divisions": 2,
        "avoided_stochastic_argmax_calls": 1,
        "avoided_where_calls": 1,
        "fallback_counts": {},
    }


def test_greedy_fast_path_fallbacks_preserve_legacy_sampler():
    cases = (
        (False, 0, (0.0,), None, (1, 3), "disabled"),
        (True, 1, (0.0,), None, (1, 3), "non_root_rank"),
        (
            True,
            0,
            (0.0, 0.0),
            None,
            (2, 3),
            "batch_size_unsupported",
        ),
        (
            True,
            0,
            (0.0,),
            "mixed",
            (1, 3),
            "mixed_batch_unsupported",
        ),
        (
            True,
            0,
            (0.7,),
            None,
            (1, 3),
            "nonzero_temperature",
        ),
        (
            True,
            0,
            (0.0,),
            None,
            (3,),
            "logits_shape_unsupported",
        ),
    )
    for enabled, rank, temperatures, batch_kind, shape, reason in cases:
        runner = make_runner(
            zero_temperature_greedy_fast_path=enabled,
        )
        runner.rank = rank
        calls = []
        prepared = object()
        runner.prepare_sample = (
            lambda seqs: calls.append(("prepare", tuple(seqs)))
            or prepared
        )
        runner.sampler = (
            lambda selected, observed_temperatures:
                calls.append(
                    (
                        "sampler",
                        selected,
                        observed_temperatures,
                    )
                )
                or SimpleNamespace(tolist=lambda: [9] * len(temperatures))
        )
        logits = _GreedyFastPathLogits(
            [[1.0, 2.0, 3.0]] * max(1, len(temperatures)),
            shape=shape,
        )
        seqs = [
            SimpleNamespace(temperature=value)
            for value in temperatures
        ]

        assert (
            runner._sample_tokens_with_optional_greedy_fast_path(
                logits,
                seqs,
                batch_kind=batch_kind,
            )
            == [9] * len(temperatures)
        )
        assert [call[0] for call in calls] == [
            "prepare",
            "sampler",
        ]
        assert (
            runner.zero_temperature_greedy_fast_path_summary()[
                "fallback_counts"
            ]
            == {reason: 1}
        )


def test_run_model_step_consumes_graph_tail_result_once():
    runner = make_runner(
        zero_temperature_greedy_fast_path=True,
        graph_resident_greedy_tail=True,
    )
    runner._record_step_logits = True
    runner.prepare_decode = lambda _seqs: (
        _TraceTensor([11]),
        _TraceTensor([32767]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = _TraceTensor([
        [0.0, 1.0, 9.0, 3.0],
    ])
    token_d2h_calls = []

    class TokenTensor:
        def __init__(self):
            self.tolist_calls = 0

        def tolist(self):
            self.tolist_calls += 1
            return [2]

    token_tensor = TokenTensor()
    observed_kwargs = {}

    def run_model(*_args, **kwargs):
        observed_kwargs.update(kwargs)
        return GraphResidentGreedyTailReplay(
            logits=logits,
            token_ids=token_tensor,
        )

    runner.run_model = run_model
    runner.graph_resident_greedy_tail = SimpleNamespace(
        mark_token_d2h=lambda: token_d2h_calls.append("d2h")
    )
    runner._sample_tokens_with_optional_greedy_fast_path = (
        lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "graph-tail result reached the host sampler"
                )
            )
        )
    )
    seq = SimpleNamespace(
        seq_id=7,
        temperature=0.0,
        hybrid_state_slot_id=-1,
        hybrid_state_generation=0,
        last_token=11,
        num_tokens=32768,
        num_completion_tokens=0,
        block_table=list(range(128)),
    )

    token_ids = runner._run_model_step(
        [seq],
        is_prefill=False,
    )

    assert token_ids == [2]
    assert token_tensor.tolist_calls == 1
    assert token_d2h_calls == ["d2h"]
    assert observed_kwargs[
        "graph_tail_temperatures"
    ] == (0.0,)
    assert observed_kwargs["graph_tail_do_sample"] is True
    assert observed_kwargs["graph_tail_batch_kind"] is None
    recorded = runner.last_step_logits()
    assert recorded is not None
    assert recorded.values == logits.values


def test_run_model_step_records_ordinary_decode_without_changing_sample():
    runner = _trace_ready_runner()
    runner.enable_spec_verify_trace_recording(True)
    runner.set_spec_verify_trace_context(
        "baseline",
        1,
        6,
    )
    runner._record_step_logits = False
    runner.kv_offload = SimpleNamespace(
        bound_generations=[1] * 128,
    )
    runner.prepare_decode = lambda _seqs: (
        _TraceTensor([11]),
        _TraceTensor([32767]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = _TraceTensor([
        [0.0, 1.0, 2.0, 3.0, 9.0, 4.0],
    ])
    runner.run_model = lambda *_args, **_kwargs: logits
    runner.prepare_sample = lambda _seqs: _TraceTensor([0.0])
    runner.sampler = lambda selected, _temperatures: (
        SimpleNamespace(
            tolist=lambda: selected.argmax(dim=-1).tolist()
        )
    )
    seq = SimpleNamespace(
        seq_id=7,
        hybrid_state_slot_id=-1,
        hybrid_state_generation=0,
        last_token=11,
        num_tokens=32768,
        num_completion_tokens=0,
        block_table=list(range(128)),
    )

    token_ids = runner._run_model_step(
        [seq],
        is_prefill=False,
    )
    rows = runner.drain_spec_verify_trace_rows()

    assert token_ids == [4]
    assert len(rows) == 1
    assert rows[0]["stage"] == "ordinary_decode"
    assert rows[0]["execution_mode"] == "decode"
    assert rows[0]["sequence_id"] == 7
    assert rows[0]["prediction_index"] == 0
    assert rows[0]["input_token_id"] == 11
    assert rows[0]["position"] == 32767
    assert rows[0]["context_length"] == 32768
    assert rows[0]["logical_block_identities"][-1] == (
        127,
        1,
    )


def test_trace_drain_does_not_change_legacy_step_logits():
    runner = _trace_ready_runner()
    runner.enable_step_logits_recording(True)
    runner.enable_spec_verify_trace_recording(True)
    runner.set_spec_verify_trace_context(
        "baseline",
        1,
        7,
    )
    runner.kv_offload = SimpleNamespace(
        bound_generations=[1] * 128,
    )
    runner.prepare_decode = lambda _seqs: (
        _TraceTensor([11]),
        _TraceTensor([32767]),
    )
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    logits = _TraceTensor([
        [0.0, 1.0, 2.0, 3.0, 9.0, 4.0],
    ])
    runner.run_model = lambda *_args, **_kwargs: logits
    runner.prepare_sample = lambda _seqs: _TraceTensor([0.0])
    runner.sampler = lambda selected, _temperatures: (
        SimpleNamespace(
            tolist=lambda: selected.argmax(dim=-1).tolist()
        )
    )
    seq = SimpleNamespace(
        seq_id=7,
        hybrid_state_slot_id=-1,
        hybrid_state_generation=0,
        last_token=11,
        num_tokens=32768,
        num_completion_tokens=0,
        block_table=list(range(128)),
    )

    assert runner._run_model_step(
        [seq],
        is_prefill=False,
    ) == [4]
    expected_logits = logits.tolist()

    assert runner.last_step_logits().tolist() == expected_logits
    assert runner.drain_spec_verify_trace_rows()
    assert runner.last_step_logits().tolist() == expected_logits
    runner.enable_spec_verify_trace_recording(False)
    assert runner.last_step_logits().tolist() == expected_logits


def test_first_target_forward_failure_leaves_no_partial_trace():
    runner = _trace_ready_runner()
    _enable_first_target_trace(runner)
    runner.run_model = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("trace forward failed"))
    seq = _trace_first_target_sequence()

    with pytest.raises(
        RuntimeError,
        match="trace forward failed",
    ):
        runner.run_spec_first_target_batch(
            (seq,),
            kv_block_identity_rows=_identity_rows((seq,)),
        )

    assert runner.drain_spec_verify_trace_rows() == ()
    assert runner.enable_spec_verify_trace_recording(False) == {
        "rank": 0,
        "enabled": False,
    }
    assert runner.drain_spec_verify_trace_rows() == ()


def test_run_clears_recorded_logits_when_sampling_is_disabled():
    runner = _make_step_logits_run_runner()
    runner.run_model = lambda *args: FakeIndexedTensor([[1.0, 2.0]])

    result = runner.run(
        [make_sequence(1)],
        is_prefill=False,
        do_sample=False,
    )

    assert result is None
    assert runner.last_step_logits() is None


def test_run_clears_recorded_logits_on_nonzero_rank():
    runner = _make_step_logits_run_runner(rank=1)
    runner.run_model = lambda *args: FakeIndexedTensor([[1.0, 2.0]])

    result = runner.run(
        [make_sequence(1)],
        is_prefill=False,
    )

    assert result is None
    assert runner.last_step_logits() is None


def test_capture_failures_are_terminal_and_reason_specific():
    runner = _make_exact_dispatch_runner()
    identity = runner._build_multi_sequence_graph_identity(
        FakeTensor([10, 20, 30, 40]),
        SimpleNamespace(
            block_tables=FakeTensor([[0, 1]] * 4),
        ),
    )
    error_type = model_runner._ExactGraphCaptureError
    runner._capture_exact_multi_sequence_graph = (
        lambda **kwargs: (
            _ for _ in ()
        ).throw(
            error_type(
                "identity_drift",
                "drift",
                retained_reserved_bytes=8192,
            )
        )
    )
    result = runner._attempt_post_step_capture(
        identity=identity,
        input_ids=FakeTensor([10, 20, 30, 40]),
        positions=FakeTensor([1, 1, 1, 1]),
        context=SimpleNamespace(
            block_tables=FakeTensor([[0, 1]] * 4),
        ),
    )
    assert result is None
    summary = runner.exact_cuda_graph_cache.summary()
    assert summary["rejected"][identity.sha256] == "identity_drift"
    assert summary["reserved_delta_bytes"] == 8192
    decision = runner.exact_cuda_graph_cache.observe_success(
        identity,
        estimated_static_bytes=4096,
    )
    assert decision.should_capture is False
    assert decision.fallback_reason == "identity_drift"


def test_capture_restores_all_scratch_slots_and_context_in_finally():
    runner = _make_exact_dispatch_runner()
    runner._capture_exact_multi_sequence_graph = (
        ModelRunner._capture_exact_multi_sequence_graph.__get__(
            runner,
            ModelRunner,
        )
    )
    runner.graph_pool = None
    runner.config.hf_config.torch_dtype = SimpleNamespace(itemsize=2)
    scratch_slots = [2048, 2304, 2560, 2816]
    runner._exact_graph_scratch_slots = (
        lambda *, batch_size: tuple(scratch_slots[:batch_size])
    )
    scratch_state = {
        slot: [slot, slot + 1]
        for slot in scratch_slots
    }
    before = copy.deepcopy(scratch_state)
    observed = {
        "snapshot_slots": None,
        "restore_slots": None,
        "restore_count": 0,
    }

    def snapshot(slots):
        observed["snapshot_slots"] = tuple(slots)
        return {
            "state": {
                slot: list(scratch_state[slot])
                for slot in slots
            }
        }

    def restore(slots, snapshot_value):
        observed["restore_slots"] = tuple(slots)
        observed["restore_count"] += 1
        for slot in slots:
            scratch_state[slot] = list(
                snapshot_value["state"][slot]
            )

    runner.snapshot_kv_slots = snapshot
    runner.restore_kv_slots = restore

    class MutatingModel:
        def __call__(self, input_ids, positions, input_embeds=None):
            del positions, input_embeds
            for slot in scratch_slots:
                scratch_state[slot][0] += 1000
            return FakeCaptureTensor(
                (input_ids.size(0), 16),
                element_size=2,
            )

        def compute_logits(self, hidden):
            return hidden

    runner.model = MutatingModel()

    class FakeGraph:
        def pool(self):
            return "pool"

    class FakeGraphContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    original_zeros = getattr(model_runner.torch, "zeros", None)
    original_cuda = model_runner.torch.cuda
    model_runner.torch.zeros = (
        lambda *shape, dtype=None, device=None: FakeCaptureTensor(
            shape[0] if len(shape) == 1 else shape,
            element_size=2 if dtype == runner.config.hf_config.torch_dtype else 4,
        )
    )
    model_runner.torch.cuda = SimpleNamespace(
        get_device_properties=original_cuda.get_device_properties,
        CUDAGraph=FakeGraph,
        graph=lambda graph, pool=None: FakeGraphContext(),
        synchronize=lambda: None,
        memory_allocated=lambda: 100,
        memory_reserved=lambda: 200,
    )
    context.set_context(
        False,
        slot_mapping=FakeTensor([0, 256, 512, 768]),
        context_lens=FakeTensor([1, 1, 1, 1]),
        block_tables=FakeTensor([[0, 1]] * 4),
    )
    identity = runner._build_multi_sequence_graph_identity(
        FakeTensor([10, 20, 30, 40]),
        context.get_context(),
    )
    try:
        entry = runner._capture_exact_multi_sequence_graph(
            identity=identity,
            input_ids=FakeTensor([10, 20, 30, 40]),
            positions=FakeTensor([1, 1, 1, 1]),
            context=context.get_context(),
        )
    finally:
        if original_zeros is None:
            delattr(model_runner.torch, "zeros")
        else:
            model_runner.torch.zeros = original_zeros
        model_runner.torch.cuda = original_cuda

    assert entry.identity_sha256 == identity.sha256
    assert observed["snapshot_slots"] == tuple(scratch_slots)
    assert observed["restore_slots"] == tuple(scratch_slots)
    assert observed["restore_count"] == 1
    assert scratch_state == before
    current = context.get_context()
    assert current.is_prefill is False
    assert current.slot_mapping is None


def test_replay_resets_context_on_success_and_exception():
    cache_module = sys.modules[
        "tinyvllm.engine.exact_cuda_graph_cache"
    ]
    for should_raise in (False, True):
        runner = _make_exact_dispatch_runner()
        context.set_context(
            False,
            slot_mapping=FakeTensor([0, 256, 512, 768]),
            context_lens=FakeTensor([1, 1, 1, 1]),
            block_tables=FakeTensor([[0, 1]] * 4),
        )
        identity = runner._build_multi_sequence_graph_identity(
            FakeTensor([10, 20, 30, 40]),
            context.get_context(),
        )

        class Graph:
            def replay(self):
                if should_raise:
                    raise RuntimeError("replay failed")

        entry = cache_module.ExactCudaGraphEntry(
            identity=identity,
            identity_sha256=identity.sha256,
            graph=Graph(),
            tensors={
                "input_ids": FakeGraphBuffer(),
                "positions": FakeGraphBuffer(),
                "slot_mapping": FakeGraphBuffer(),
                "context_lens": FakeGraphBuffer(),
                "block_tables": FakeGraphBuffer(),
                "outputs": FakeTensor([[1], [2], [3], [4]]),
            },
            static_bytes=4096,
            capture_duration_ns=100,
            allocated_delta_bytes=0,
            reserved_delta_bytes=1024,
        )
        for _ in range(3):
            runner.exact_cuda_graph_cache.observe_success(
                identity,
                estimated_static_bytes=4096,
            )
        runner.exact_cuda_graph_cache.commit_capture(entry)
        try:
            runner._replay_exact_multi_sequence_graph(
                entry,
                input_ids=FakeTensor([10, 20, 30, 40]),
                positions=FakeTensor([1, 1, 1, 1]),
                context=context.get_context(),
            )
        except RuntimeError:
            assert should_raise
        else:
            assert not should_raise
        current = context.get_context()
        assert current.is_prefill is False
        assert current.slot_mapping is None
        if should_raise:
            assert (
                runner.exact_cuda_graph_cache.summary()["rejected"][
                    identity.sha256
                ]
                == "replay_disabled"
            )


def test_replay_failure_publishes_terminal_event_before_reraising():
    runner = _make_exact_dispatch_runner()
    for _ in range(3):
        _run_exact_decode(runner)
    ready_entry = next(
        iter(runner.exact_cuda_graph_cache.ready_entries.values())
    )

    class RaisingGraph:
        def replay(self):
            raise RuntimeError("replay failed")

    ready_entry.graph = RaisingGraph()
    try:
        _run_exact_decode(runner)
    except RuntimeError as exc:
        assert "replay failed" in str(exc)
    else:
        raise AssertionError("replay failure must not rerun eager")

    event = runner.cuda_graph_dispatch_observation()
    assert tuple(event) == model_runner.DISPATCH_EVENT_FIELDS
    assert event["dispatch"] == "graph"
    assert event["cache_state"] == "rejected"
    assert event["fallback_reason"] == "replay_disabled"
    assert event["graph_identity_sha256"] == ready_entry.identity_sha256


def main():
    tests = (
        test_init_prepares_cuda_graph_dispatch_state_before_warmup,
        test_prefill_window_reserves_current_write_blocks_without_mutating_decode_window,
        test_prepare_spec_verify_installs_reference_context,
        test_prepare_spec_verify_batch_flattens_homogeneous_rows_once,
        test_step_logits_recording_accessor_is_default_off_and_returns_clone,
        test_snapshot_kv_slots_uses_physical_block_and_offset_indices,
        test_snapshot_kv_slots_rejects_empty_or_quantized_requests,
        test_prepare_spec_verify_rejects_nonconsecutive_slots_before_upload,
        test_every_unsupported_feature_fails_closed,
        test_multi_sequence_is_allowed_but_invalid_modes_fail,
        test_spec_verify_run_model_uses_eager_and_keeps_all_rows,
        test_run_spec_verify_batch_uses_one_forward_and_splits_rows,
        test_run_spec_verify_batch_worker_executes_forward_without_result,
        test_run_spec_verify_batch_resets_context_after_forward_failure,
        test_run_spec_first_target_batch_uses_one_forward_and_orders_rows,
        test_run_offload_spec_first_target_batch_does_not_require_residency_ticket,
        test_run_spec_first_target_batch_worker_executes_forward_without_result,
        test_run_spec_first_target_batch_resets_context_after_forward_failure,
        test_run_spec_first_target_batch_rejects_non_greedy_before_prepare,
        test_hybrid_state_spec_verify_fails_closed_before_forward,
        test_multi_sequence_decode_uses_eager_instead_of_cuda_graph,
        test_single_sequence_decode_uses_eager_when_legacy_graph_state_is_absent,
        test_single_sequence_decode_still_replays_cuda_graph,
        test_replay_aware_decode_summary_delegates_to_arena,
        test_replay_aware_decode_preparation_lands_exact_batch_one,
        test_replay_aware_decode_preparation_fails_closed_when_disabled,
        test_prelanded_single_sequence_replay_skips_copy_and_zero,
        test_exact_graph_capacity_reserves_scheduler_invisible_scratch,
        test_spec_verify_scratch_capacity_covers_worst_row_offset_without_padding,
        test_decode_and_spec_verify_scratch_partitions_are_disjoint,
        test_exact_burst_capacity_adds_one_scheduler_invisible_block,
        test_exact_burst_scratch_is_reported_by_capacity_snapshot,
        test_exact_burst_capability_is_fail_closed_and_json_safe,
        test_model_runner_exact_burst_delegates_once_with_padded_block_table,
        test_model_runner_split_phase_delegates_to_k8_mailbox_backend,
        test_model_runner_split_phase_releases_or_aborts_owned_generation,
        test_model_runner_exact_burst_materializes_only_in_lazy_factory,
        test_model_runner_invalidates_both_burst_graphs,
        test_model_runner_exact_burst_replay_enters_inference_mode,
        test_model_runner_exact_burst_correctness_capture_enters_inference_mode,
        test_model_runner_exact_burst_rejects_wrong_sequence_before_graph,
        test_model_runner_exact_burst_capture_owns_static_state_and_pool,
        test_scratch_blocks_are_above_scheduler_visible_range,
        test_feature_enabled_startup_captures_only_batch_one,
        test_feature_disabled_startup_inventory_is_unchanged,
        test_exact_graph_identity_and_static_byte_estimate_are_exact,
        test_three_successful_eager_steps_capture_post_step_and_fourth_replays,
        test_exact_ready_entry_never_rounds_batch_or_page_table_width,
        test_multi_sequence_graph_guards_report_exact_fallback_reasons,
        test_non_allowlisted_and_invalid_identity_fail_closed,
        test_dispatch_event_schema_is_complete_and_ordered,
        test_every_fallback_reason_and_graph_hit_use_closed_event_schema,
        test_run_hashes_canonical_sorted_sequence_ids_before_dispatch,
        test_run_records_selected_rank_zero_sampling_logits_before_sampler,
        test_zero_temperature_greedy_fast_path_config_is_fail_closed,
        test_graph_resident_greedy_tail_config_is_fail_closed,
        test_exact_greedy_decode_burst_config_is_strict_and_default_off,
        test_greedy_fast_path_uses_exact_float32_argmax,
        test_greedy_fast_path_fallbacks_preserve_legacy_sampler,
        test_graph_tail_capture_binds_batch_one_static_output,
        test_capture_cudagraph_initializes_exact_burst_after_generation,
        test_graph_tail_decides_before_transformer_replay_and_skips_logits,
        test_graph_tail_ineligible_step_preserves_external_logits_path,
        test_graph_tail_replay_failure_never_falls_back_or_replays_twice,
        test_run_model_step_consumes_graph_tail_result_once,
        test_run_clears_recorded_logits_when_sampling_is_disabled,
        test_run_clears_recorded_logits_on_nonzero_rank,
        test_capture_failures_are_terminal_and_reason_specific,
        test_capture_restores_all_scratch_slots_and_context_in_finally,
        test_replay_resets_context_on_success_and_exception,
        test_replay_failure_publishes_terminal_event_before_reraising,
    )
    for test in tests:
        context.reset_context()
        test()
    print("model runner spec_verify tests passed")


if __name__ == "__main__":
    main()
