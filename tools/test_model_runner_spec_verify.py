"""Dependency-light tests for ModelRunner native verifier preparation."""

from __future__ import annotations

import __future__
import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

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
    tinyvllm_package.__path__ = []
    for package_name in (
        "tinyvllm.speculative",
        "tinyvllm.engine",
        "tinyvllm.models",
        "tinyvllm.utils",
        "tinyvllm.layers",
    ):
        package = _install_module(package_name)
        package.__path__ = []

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
    _install_module("flash_attn", __version__="2.6.3")

    _install_module("tinyvllm.config", Config=type("Config", (), {}))
    _install_module(
        "tinyvllm.engine.sequence",
        Sequence=type("Sequence", (), {}),
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
    return model_runner, context_module


model_runner, context = _load_model_runner_module()
ModelRunner = model_runner.ModelRunner


def make_runner(**overrides):
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    runner.world_size = 1
    runner.kv_offload = None
    runner.enforce_eager = False
    config = {
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "quest_top_k_blocks": -1,
        "am_compact_blocks": 0,
        "kv_cartridge_blocks": 0,
        "chunked_prefill_mixed_batch": False,
        "cpu_offload": False,
        "multi_sequence_cuda_graphs": False,
        "multi_sequence_cuda_graph_batch_allowlist": (2, 4, 8),
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
    runner._list_to_cuda = (
        lambda data, name, dtype: FakeTensor(list(data))
    )
    runner.prepare_block_tables_from_rows = (
        lambda rows, name="block_tables": FakeTensor(
            [list(row) for row in rows]
        )
    )
    return runner


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
    assert positions.values == [53, 54, 55]
    assert metadata.query_len == 3
    assert metadata.logical_slots == (52, 53, 54)
    assert metadata.physical_slots == (52, 53, 54)
    assert metadata.context_len == 55
    assert current.mode == "spec_verify"
    assert current.context_lens.values == [55]
    assert current.block_tables.values == [[0]]
    assert current.flash_attn_num_splits == 16


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
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_prefill": True,
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


def test_multi_sequence_nonlinear_and_nongreedy_fail():
    runner = make_runner()
    invalid = (
        dict(
            seq_count=2,
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
            "unsupported_mode",
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
        event = runner.cuda_graph_dispatch_observation()
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
        SimpleNamespace(seq_id=9),
        SimpleNamespace(seq_id=2),
        SimpleNamespace(seq_id=5),
    ]
    runner.run(seqs, is_prefill=False, do_sample=False)
    expected = hashlib.sha256(
        json.dumps(
            [2, 5, 9],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert observed["request_ids_hash"] == expected


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
        test_prepare_spec_verify_installs_reference_context,
        test_snapshot_kv_slots_uses_physical_block_and_offset_indices,
        test_snapshot_kv_slots_rejects_empty_or_quantized_requests,
        test_prepare_spec_verify_rejects_nonconsecutive_slots_before_upload,
        test_every_unsupported_feature_fails_closed,
        test_multi_sequence_nonlinear_and_nongreedy_fail,
        test_spec_verify_run_model_uses_eager_and_keeps_all_rows,
        test_multi_sequence_decode_uses_eager_instead_of_cuda_graph,
        test_single_sequence_decode_still_replays_cuda_graph,
        test_exact_graph_capacity_reserves_scheduler_invisible_scratch,
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
