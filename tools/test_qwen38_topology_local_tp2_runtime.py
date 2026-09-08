import __future__
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"

for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        label,
        dtype="torch.bfloat16",
        device="cuda:0",
        floating=True,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.label = label
        self.dtype = dtype
        self.device = device
        self._floating = floating
        self._contiguous = contiguous

    def is_floating_point(self):
        return self._floating

    def narrow(self, dim, start, length):
        shape = list(self.shape)
        shape[dim] = length
        return FakeTensor(
            shape,
            label=f"{self.label}.narrow({dim},{start},{length})",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=False,
        )

    def contiguous(self):
        return FakeTensor(
            self.shape,
            label=f"{self.label}.contiguous",
            dtype=self.dtype,
            device=self.device,
            floating=self._floating,
            contiguous=True,
        )

    def to(self, *, dtype=None, device=None):
        return FakeTensor(
            self.shape,
            label=f"{self.label}.to",
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            floating=self._floating,
            contiguous=self._contiguous,
        )

    def numel(self):
        result = 1
        for width in self.shape:
            result *= width
        return result

    def element_size(self):
        return {
            "torch.bfloat16": 2,
            "torch.float32": 4,
        }[self.dtype]


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
fake_torch.bfloat16 = "torch.bfloat16"
fake_torch.float32 = "torch.float32"
fake_torch.uint8 = "torch.uint8"


class FakeModule:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


fake_torch.nn = SimpleNamespace(Module=FakeModule)


def _cat(tensors, dim=0):
    tensors = tuple(tensors)
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return FakeTensor(
        shape,
        label="cat(" + ",".join(tensor.label for tensor in tensors) + ")",
        dtype=tensors[0].dtype,
        device=tensors[0].device,
        floating=all(tensor._floating for tensor in tensors),
        contiguous=False,
    )


fake_torch.cat = _cat


@pytest.fixture(autouse=True)
def _isolate_fake_torch():
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        code = compile(
            path.read_text(),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        return module
    except Exception:
        sys.modules.pop(name, None)
        raise


def _load_real_config_class():
    module_name = "tinyvllm_config_topology_local_tp2_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=64,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original_transformers = sys.modules.get("transformers")
    try:
        sys.modules["transformers"] = fake_transformers
        return _load_source_module(module_name, CONFIG_PATH).Config
    finally:
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers
        sys.modules.pop(module_name, None)


def _build_config(**overrides):
    Config = _load_real_config_class()
    model_directory = tempfile.TemporaryDirectory()
    config = Config(
        model=model_directory.name,
        max_num_batched_tokens=4096,
        **overrides,
    )
    config._test_model_directory = model_directory
    return config


def test_topology_local_tp2_config_is_default_off_and_strict_bool():
    Config = _load_real_config_class()

    assert (
        Config.__dataclass_fields__[
            "qwen38_topology_local_tp2_islands"
        ].default
        is False
    )
    with pytest.raises(
        ValueError,
        match=(
            "^qwen38_topology_local_tp2_islands must be a bool$"
        ),
    ):
        _build_config(qwen38_topology_local_tp2_islands=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"tensor_parallel_size": 2, "enforce_eager": True},
            "requires tensor_parallel_size 4",
        ),
        (
            {"tensor_parallel_size": 4, "enforce_eager": False},
            "requires eager execution",
        ),
    ),
)
def test_topology_local_tp2_config_rejects_wrong_execution_contract(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _build_config(
            qwen38_topology_local_tp2_islands=True,
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"qwen35_mtp_enabled": True},
        {"autoregressive_draft_enabled": True},
        {"multi_sequence_cuda_graphs": True},
        {"prefill_cuda_graphs": True},
        {"cpu_offload": True},
        {"kv_offload_mvp0": True},
        {"exact_greedy_decode_burst": True},
        {"graph_resident_greedy_tail": True},
        {"quantization": "int8"},
        {"kv_quant_bits": 8},
        {"act_quant_bits": 8},
        {"smoothquant_scale_path": "non-null"},
        {"quest_top_k_blocks": 1},
        {"kv_cartridge_blocks": 1},
        {"am_compact_blocks": 1},
    ),
)
def test_topology_local_tp2_config_rejects_incompatible_modes(
    overrides,
):
    with pytest.raises(ValueError, match="incompatible"):
        _build_config(
            tensor_parallel_size=4,
            enforce_eager=True,
            qwen38_topology_local_tp2_islands=True,
            **overrides,
        )


class FakeDistributed:
    def __init__(self):
        self.calls = []

    def new_group(self, *, ranks):
        group = ("group", tuple(ranks))
        self.calls.append(tuple(ranks))
        return group


def test_pair_context_creates_all_groups_in_global_order():
    from tinyvllm.engine.topology_local_tp2_island import (
        TopologyLocalTP2PairMap,
        create_topology_local_tp2_pair_context,
    )

    pair_map = TopologyLocalTP2PairMap(((0, 1), (2, 3)))
    for global_rank in range(4):
        distributed = FakeDistributed()
        context = create_topology_local_tp2_pair_context(
            distributed,
            global_rank=global_rank,
            world_size=4,
            pair_map=pair_map,
        )

        assert distributed.calls == [(0, 1), (2, 3)]
        assert context.identity.global_rank == global_rank
        assert context.identity.pair_id == global_rank // 2
        assert context.identity.logical_rank == global_rank % 2
        assert context.pair_group == (
            "group",
            pair_map.pair_groups[global_rank // 2],
        )
        assert context.all_pair_groups == (
            ("group", (0, 1)),
            ("group", (2, 3)),
        )


@pytest.mark.parametrize("world_size", (1, 2, 3, 5, True))
def test_pair_context_rejects_non_tp4_world_size(world_size):
    from tinyvllm.engine.topology_local_tp2_island import (
        TopologyLocalTP2PairMap,
        create_topology_local_tp2_pair_context,
    )

    with pytest.raises(
        ValueError,
        match="requires world_size 4",
    ):
        create_topology_local_tp2_pair_context(
            FakeDistributed(),
            global_rank=0,
            world_size=world_size,
            pair_map=TopologyLocalTP2PairMap(((0, 1), (2, 3))),
        )


class Qwen35LinearAttentionShell:
    pass


def _fake_linear_attention(**overrides):
    layer = Qwen35LinearAttentionShell()
    values = {
        "local_key_heads": 4,
        "local_value_heads": 12,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "norm_eps": 1e-6,
        "in_proj_qkv": SimpleNamespace(
            weight=FakeTensor((10240, 5120), label="qkv"),
            quant_method=None,
        ),
        "in_proj_z": SimpleNamespace(
            weight=FakeTensor((6144, 5120), label="z"),
            quant_method=None,
        ),
        "in_proj_b": SimpleNamespace(
            weight=FakeTensor((48, 5120), label="b"),
            quant_method=None,
        ),
        "in_proj_a": SimpleNamespace(
            weight=FakeTensor((48, 5120), label="a"),
            quant_method=None,
        ),
        "out_proj": SimpleNamespace(
            weight=FakeTensor((5120, 1536), label="out-quarter"),
            prefill_weight=FakeTensor((5120, 6144), label="out-full"),
            accumulation_weight=FakeTensor(
                (5120, 1536),
                label="out-quarter-fp32",
                dtype="torch.float32",
            ),
            quant_method=None,
            tp_size=4,
        ),
        "logical_tp2_conv_weight": FakeTensor(
            (10240, 4),
            label="conv-full",
        ),
        "logical_tp2_A_log": FakeTensor(
            (48,),
            label="A-log-full",
            dtype="torch.float32",
        ),
        "logical_tp2_dt_bias": FakeTensor(
            (48,),
            label="dt-bias-full",
        ),
        "norm_weight": FakeTensor((128,), label="norm"),
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(layer, name, value)
    return layer


def _load_linear_attention_module():
    module_path = (
        ROOT
        / "tinyvllm/layers/"
        "qwen38_topology_local_tp2_linear_attention.py"
    )
    assert module_path.is_file(), (
        "Qwen3.8 topology-local TP2 runtime layer module is missing"
    )
    sys.modules["torch"] = fake_torch
    return _load_source_module(
        "qwen38_topology_local_tp2_linear_attention_under_test",
        module_path,
    )


def test_logical_tp2_view_selects_exact_checkpoint_halves():
    runtime = _load_linear_attention_module()
    runtime._tensor_digest = lambda tensor: "a" * 64
    layer = _fake_linear_attention()

    view = runtime.build_logical_tp2_linear_attention_view(
        layer,
        logical_rank=1,
        pair_group="pair-b",
    )

    assert view.logical_parallel_size == 2
    assert view.logical_rank == 1
    assert view.key_head_range == (8, 16)
    assert view.value_head_range == (24, 48)
    assert view.output_input_range == (3072, 6144)
    assert tuple(
        tensor.shape for tensor in view.qkv_weight_segments
    ) == (
        (1024, 5120),
        (1024, 5120),
        (3072, 5120),
    )
    assert view.z_weight.shape == (6144, 5120)
    assert view.output_accumulation_weight.shape == (5120, 3072)
    assert view.output_accumulation_weight.dtype == "torch.float32"
    assert runtime.build_layer_parameter_identity(
        layer,
        view,
    )["checkpoint_reconstruction_match"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda layer: object(),
        lambda layer: setattr(
            layer.in_proj_qkv,
            "quant_method",
            object(),
        ) or layer,
        lambda layer: setattr(
            layer.out_proj,
            "prefill_weight",
            None,
        ) or layer,
        lambda layer: setattr(
            layer,
            "local_key_heads",
            8,
        ) or layer,
    ),
)
def test_logical_tp2_view_rejects_invalid_layer_contract(mutation):
    runtime = _load_linear_attention_module()

    with pytest.raises(ValueError):
        runtime.build_logical_tp2_linear_attention_view(
            mutation(_fake_linear_attention()),
            logical_rank=0,
            pair_group="pair-a",
        )


@pytest.mark.parametrize("logical_rank", (-1, 2, True))
def test_logical_tp2_view_rejects_invalid_logical_rank(logical_rank):
    runtime = _load_linear_attention_module()

    with pytest.raises(ValueError, match="logical_rank"):
        runtime.build_logical_tp2_linear_attention_view(
            _fake_linear_attention(),
            logical_rank=logical_rank,
            pair_group="pair-a",
        )


def test_candidate_view_registration_is_closed_after_warmup():
    runtime = _load_linear_attention_module()
    lifecycle = runtime.CandidateSetupLifecycle()
    lifecycle.mark_warmup_started()

    with pytest.raises(RuntimeError, match="before warmup"):
        lifecycle.register_candidate_view(
            runtime.build_logical_tp2_linear_attention_view(
                _fake_linear_attention(),
                logical_rank=0,
                pair_group="pair-a",
            )
        )


def test_release_global_tp4_decode_accumulation_drops_last_owner():
    runtime = _load_linear_attention_module()
    layer = _fake_linear_attention()
    reference = weakref.ref(layer.out_proj.accumulation_weight)

    receipt = runtime.release_global_tp4_decode_accumulation(layer)
    gc.collect()

    assert receipt == {
        "released_bytes": 5120 * 1536 * 4,
        "released": True,
    }
    assert layer.out_proj.accumulation_weight is None
    assert reference() is None


class _BaselineMixer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def _wrapper_view():
    return SimpleNamespace(
        pair_group="pair-a",
        logical_parallel_size=2,
    )


def test_pair_local_wrapper_dispatches_prefill_then_decode_and_counts():
    runtime = _load_linear_attention_module()
    baseline_result = object()
    baseline = _BaselineMixer(baseline_result)
    reductions = []
    wrapper = runtime.Qwen38TopologyLocalTP2LinearAttention(
        baseline=baseline,
        candidate_view=_wrapper_view(),
        pair_reduce=lambda output, group: reductions.append(
            (output, group)
        ),
    )
    hidden = FakeTensor((3, 5120), label="prefill")
    convolution = object()
    recurrent = object()

    assert wrapper.phase == "tp4_prefill"
    assert wrapper(hidden, convolution, recurrent) is baseline_result
    assert baseline.calls == [(hidden, convolution, recurrent)]

    projected = SimpleNamespace(
        gate=object(),
        next_convolution=object(),
    )
    core = object()
    next_recurrent = object()
    local = FakeTensor(
        (1, 5120),
        label="local-output",
        dtype="torch.float32",
    )
    wrapper._project_logical_tp2 = (
        lambda hidden_states, convolution_state: projected
    )
    wrapper._run_delta = (
        lambda projection, recurrent_state, *, token_count: (
            core,
            next_recurrent,
        )
    )
    wrapper._output_projection = lambda value, gate: local
    wrapper.activate_tp2_decode()

    output, next_convolution, actual_recurrent = wrapper(
        FakeTensor((1, 5120), label="decode"),
        object(),
        object(),
    )

    assert output.dtype == "torch.bfloat16"
    assert next_convolution is projected.next_convolution
    assert actual_recurrent is next_recurrent
    assert reductions == [(local, "pair-a")]
    snapshot = wrapper.telemetry_snapshot()
    assert snapshot["tp4_prefill_calls"] == 1
    assert snapshot["tp2_decode_calls"] == 1
    assert snapshot["recurrent_token_one_calls"] == 1
    assert snapshot["short_chunk_calls"] == 0
    assert snapshot["global_tp4_decode_all_reduce_calls"] == 0
    assert snapshot["pair_local_all_reduce_calls"] == 1


@pytest.mark.parametrize(
    ("token_count", "expected"),
    (
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 64),
        (64, 64),
    ),
)
def test_candidate_chunk_size_preserves_short_exact_lengths(
    token_count,
    expected,
):
    runtime = _load_linear_attention_module()

    assert runtime.candidate_gated_delta_chunk_size(
        token_count
    ) == expected


def test_pair_local_wrapper_cannot_reverse_decode_phase():
    runtime = _load_linear_attention_module()
    wrapper = runtime.Qwen38TopologyLocalTP2LinearAttention(
        baseline=_BaselineMixer(object()),
        candidate_view=_wrapper_view(),
        pair_reduce=lambda output, group: None,
    )

    wrapper.activate_tp2_decode()

    with pytest.raises(RuntimeError, match="already active"):
        wrapper.activate_tp2_decode()
    with pytest.raises(AttributeError):
        wrapper.phase = "tp4_prefill"


class _DigestTensor:
    dtype = "torch.bfloat16"

    def __init__(self, payload):
        self.payload = payload

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        assert dtype == "torch.uint8"
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.payload


def test_output_digest_is_bitwise_and_bf16_only():
    runtime = _load_linear_attention_module()

    first = runtime.output_digest(_DigestTensor(b"\x00\x01"))
    assert runtime.output_digest(_DigestTensor(b"\x00\x01")) == first
    assert runtime.output_digest(_DigestTensor(b"\x00\x00")) != first
    invalid = _DigestTensor(b"\x00\x01")
    invalid.dtype = "torch.float32"
    with pytest.raises(ValueError, match="requires BF16"):
        runtime.output_digest(invalid)
