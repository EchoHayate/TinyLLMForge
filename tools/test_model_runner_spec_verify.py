"""Dependency-light tests for ModelRunner native verifier preparation."""

from __future__ import annotations

import __future__
import importlib.util
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


class FakeCaptureTensor(FakeGraphBuffer):
    def __init__(self, shape):
        super().__init__(None)
        self.shape = (
            (shape,)
            if isinstance(shape, int)
            else tuple(shape)
        )

    def __getitem__(self, index):
        return self

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]


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
    }
    config.update(overrides)
    runner.config = SimpleNamespace(**config)
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
    )
    for test in tests:
        context.reset_context()
        test()
    print("model runner spec_verify tests passed")


if __name__ == "__main__":
    main()
