import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter = load_module(
    "tinyvllm.engine.qwen35_hybrid_state",
    "tinyvllm/engine/qwen35_hybrid_state.py",
)
build_qwen35_hybrid_state_layout = (
    adapter.build_qwen35_hybrid_state_layout
)

CANONICAL_LINEAR_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10,
    12, 13, 14, 16, 17, 18, 20, 21, 22,
)
CANONICAL_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(24)
)


def make_config(**overrides):
    values = {
        "num_hidden_layers": 24,
        "layer_types": CANONICAL_LAYER_TYPES,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def component_map(layout):
    return {
        (component.layer_index, component.role): component
        for component in layout.components
    }


def assert_rejected(config, **kwargs):
    try:
        build_qwen35_hybrid_state_layout(
            config,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            dtype=kwargs.get("dtype", torch.bfloat16),
            recurrent_dtype=kwargs.get("recurrent_dtype"),
            speculative_tokens=kwargs.get("speculative_tokens", 1),
        )
    except (TypeError, ValueError):
        return
    raise AssertionError(
        f"invalid Qwen3.5 layout config was accepted: {config}, {kwargs}"
    )


def test_canonical_tp1_bf16_layout():
    layout = build_qwen35_hybrid_state_layout(
        make_config(),
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )
    assert len(layout.components) == 36
    assert tuple(sorted({
        component.layer_index for component in layout.components
    })) == CANONICAL_LINEAR_LAYERS
    components = component_map(layout)
    for layer_index in CANONICAL_LINEAR_LAYERS:
        assert components[
            (layer_index, "linear_convolution")
        ].shape == (6144, 4)
        assert components[
            (layer_index, "linear_recurrent")
        ].shape == (16, 128, 128)
    assert layout.bytes_per_slot == 10_321_920


def test_tp2_is_rank_local_and_half_size():
    tp1 = build_qwen35_hybrid_state_layout(
        make_config(),
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )
    tp2 = build_qwen35_hybrid_state_layout(
        make_config(),
        tensor_parallel_size=2,
        dtype=torch.bfloat16,
    )
    components = component_map(tp2)
    assert components[(0, "linear_convolution")].shape == (3072, 4)
    assert components[(0, "linear_recurrent")].shape == (8, 128, 128)
    assert tp2.bytes_per_slot * 2 == tp1.bytes_per_slot


def test_mixed_bf16_convolution_fp32_recurrent_layout():
    layout = build_qwen35_hybrid_state_layout(
        make_config(),
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
    )
    components = component_map(layout)
    for layer_index in CANONICAL_LINEAR_LAYERS:
        assert components[
            (layer_index, "linear_convolution")
        ].dtype == torch.bfloat16
        assert components[
            (layer_index, "linear_recurrent")
        ].dtype == torch.float32
    recurrent_bytes = (
        18 * 16 * 128 * 128 * torch.float32.itemsize
    )
    convolution_bytes = (
        18 * 6144 * 4 * torch.bfloat16.itemsize
    )
    assert layout.bytes_per_slot == recurrent_bytes + convolution_bytes
    assert layout.fingerprint != build_qwen35_hybrid_state_layout(
        make_config(),
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    ).fingerprint


def test_fp32_text_config_and_speculative_width():
    text_config = make_config()
    wrapped = SimpleNamespace(text_config=text_config)
    bf16 = build_qwen35_hybrid_state_layout(
        wrapped,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )
    fp32 = build_qwen35_hybrid_state_layout(
        wrapped,
        tensor_parallel_size=1,
        dtype=torch.float32,
        speculative_tokens=3,
    )
    assert fp32.bytes_per_slot > 2 * bf16.bytes_per_slot
    assert component_map(fp32)[
        (0, "linear_convolution")
    ].shape == (6144, 6)
    recurrent_bytes = (
        18 * 16 * 128 * 128 * torch.float32.itemsize
    )
    convolution_bytes = (
        18 * 6144 * 6 * torch.float32.itemsize
    )
    assert fp32.bytes_per_slot == recurrent_bytes + convolution_bytes


def test_rejects_missing_invalid_and_ambiguous_fields():
    for field in (
        "num_hidden_layers",
        "layer_types",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ):
        values = vars(make_config()).copy()
        del values[field]
        assert_rejected(SimpleNamespace(**values))
    assert_rejected(make_config(num_hidden_layers=True))
    assert_rejected(make_config(linear_key_head_dim=0))
    assert_rejected(make_config(layer_types=CANONICAL_LAYER_TYPES[:-1]))
    invalid_types = list(CANONICAL_LAYER_TYPES)
    invalid_types[0] = "gated_delta"
    assert_rejected(make_config(layer_types=tuple(invalid_types)))
    assert_rejected(make_config(
        layer_types=("full_attention",) * 24,
    ))


def test_rejects_invalid_dtype_tp_and_speculative_width():
    assert_rejected(make_config(), dtype=torch.int8)
    assert_rejected(
        make_config(),
        dtype=torch.bfloat16,
        recurrent_dtype=torch.int8,
    )
    for tensor_parallel_size in (0, 9, True):
        assert_rejected(
            make_config(),
            tensor_parallel_size=tensor_parallel_size,
        )
    assert_rejected(
        make_config(linear_num_key_heads=15),
        tensor_parallel_size=2,
    )
    assert_rejected(
        make_config(linear_num_value_heads=15),
        tensor_parallel_size=2,
    )
    assert_rejected(
        make_config(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=1,
            linear_value_head_dim=1,
        ),
        tensor_parallel_size=3,
    )
    for speculative_tokens in (0, -1, True):
        assert_rejected(
            make_config(),
            speculative_tokens=speculative_tokens,
        )


if __name__ == "__main__":
    test_canonical_tp1_bf16_layout()
    test_tp2_is_rank_local_and_half_size()
    test_mixed_bf16_convolution_fp32_recurrent_layout()
    test_fp32_text_config_and_speculative_width()
    test_rejects_missing_invalid_and_ambiguous_fields()
    test_rejects_invalid_dtype_tp_and_speculative_width()
    print("qwen35 hybrid state layout tests passed")
