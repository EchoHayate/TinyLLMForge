from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid_module = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
layout_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_state",
    "tinyvllm/engine/qwen35_hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)
decoder_module = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
stack_module = _load_module(
    "tinyvllm.layers.qwen35_packed_layer_stack",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
)
root_module = _load_module(
    "tinyvllm.models.qwen35_packed",
    "tinyvllm/models/qwen35_packed.py",
)
owner_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_model_owner",
    "tinyvllm/engine/qwen35_hybrid_model_owner.py",
)
factory_module = _load_module(
    "tinyvllm.models.qwen35_factory",
    "tinyvllm/models/qwen35_factory.py",
)

HybridStateLease = hybrid_module.HybridStateLease
HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
build_qwen35_hybrid_state_layout = (
    layout_module.build_qwen35_hybrid_state_layout
)
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedForCausalLM = root_module.Qwen35PackedForCausalLM
build_qwen35_hybrid_model_owner = (
    owner_module.build_qwen35_hybrid_model_owner
)
Qwen35PackedModelAssembly = factory_module.Qwen35PackedModelAssembly
assemble_qwen35_packed_model = (
    factory_module.assemble_qwen35_packed_model
)


def _config(**overrides):
    values = {
        "num_hidden_layers": 3,
        "layer_types": (
            "linear_attention",
            "full_attention",
            "linear_attention",
        ),
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _pool(config=None):
    config = _config() if config is None else config
    layout = build_qwen35_hybrid_state_layout(
        config,
        tensor_parallel_size=1,
        dtype=torch.float32,
    )
    return HybridStateTensorPool(layout, 2, "cpu")


class _Embedding(nn.Module):
    def forward(self, input_ids):
        values = input_ids.to(torch.float32)
        return torch.stack((values, values + 1, values + 2), dim=-1)


class _Identity(nn.Module):
    def forward(self, tensor):
        return tensor


class _Head(nn.Module):
    def forward(self, hidden):
        return torch.stack((
            hidden.sum(dim=-1),
            hidden[:, 0],
        ), dim=-1)


class _Linear(nn.Module):
    def forward(self, hidden, convolution, recurrent):
        delta = hidden.sum().to(convolution.dtype)
        return hidden * 0.25, convolution + delta, recurrent + delta


class _Full(nn.Module):
    def forward(self, positions, hidden):
        return hidden * 0.5


def _decoder(block_type):
    return Qwen35DecoderLayerShell(
        block_type=block_type,
        input_layernorm=_Identity(),
        post_attention_layernorm=_Identity(),
        mlp=nn.Sequential(),
        full_attention=_Full() if block_type == "full_attention" else None,
        linear_attention=_Linear() if block_type == "linear_attention" else None,
    )


def _snapshot(pool):
    return {
        key: tensor.clone()
        for key, tensor in pool._tensors.items()
    }


def _assemble(config=None, pool=None, callback=None):
    config = _config() if config is None else config
    pool = _pool(config) if pool is None else pool
    calls = []

    def default_callback(layer_index, block_type, adapter):
        calls.append((layer_index, block_type, adapter))
        return _decoder(block_type)

    assembly = assemble_qwen35_packed_model(
        config,
        pool=pool,
        embed_tokens=_Embedding(),
        final_norm=_Identity(),
        lm_head=_Head(),
        build_decoder_layer=(
            default_callback if callback is None else callback
        ),
    )
    return pool, calls, assembly


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def test_assembly_preserves_exact_topology_identity_and_storage():
    pool = _pool()
    storage = {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    }
    values = _snapshot(pool)

    _, calls, assembly = _assemble(pool=pool)

    assert isinstance(assembly, Qwen35PackedModelAssembly)
    assert isinstance(assembly.model, Qwen35PackedForCausalLM)
    assert assembly.model.layer_stack is assembly.layer_stack
    assert (
        assembly.layer_stack.state_transaction
        is assembly.state_transaction
    )
    assert assembly.state_transaction.adapters is assembly.adapters
    assert assembly.state_transaction.pool is pool
    assert assembly.pool is pool
    assert [call[:2] for call in calls] == [
        (0, "linear_attention"),
        (1, "full_attention"),
        (2, "linear_attention"),
    ]
    assert calls[0][2] is assembly.adapters[0]
    assert calls[1][2] is None
    assert calls[2][2] is assembly.adapters[1]
    assert tuple(adapter.layer_index for adapter in assembly.adapters) == (
        0,
        2,
    )
    assert all(adapter.pool is pool for adapter in assembly.adapters)
    assert {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    } == storage
    for key, tensor in values.items():
        torch.testing.assert_close(pool._tensors[key], tensor)


def test_assembly_rejects_pool_topology_mismatch():
    config = _config()
    layout = build_qwen35_hybrid_state_layout(
        _config(layer_types=(
            "linear_attention",
            "linear_attention",
            "full_attention",
        )),
        tensor_parallel_size=1,
        dtype=torch.float32,
    )
    pool = HybridStateTensorPool(layout, 2, "cpu")
    _expect_error(
        lambda: _assemble(config=config, pool=pool),
        ValueError,
        "pool linear layer indices",
    )

    missing_role_layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (12, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            2,
            "linear_convolution",
            (12, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            2,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))
    _expect_error(
        lambda: _assemble(
            config=config,
            pool=HybridStateTensorPool(
                missing_role_layout,
                2,
                "cpu",
            ),
        ),
        ValueError,
        "convolution and recurrent",
    )

    extra_full_layout = HybridStateLayout((
        *build_qwen35_hybrid_state_layout(
            config,
            tensor_parallel_size=1,
            dtype=torch.float32,
        ).components,
        HybridStateComponentSpec(
            1,
            "linear_convolution",
            (12, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            1,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))
    _expect_error(
        lambda: _assemble(
            config=config,
            pool=HybridStateTensorPool(
                extra_full_layout,
                2,
                "cpu",
            ),
        ),
        ValueError,
        "pool linear layer indices",
    )


def test_assembly_accepts_text_config_and_rejects_pool_subclass():
    config = _config()
    wrapped = types.SimpleNamespace(text_config=config)
    pool = _pool(config)

    _, calls, assembly = _assemble(config=wrapped, pool=pool)

    assert assembly.pool is pool
    assert [call[:2] for call in calls] == [
        (0, "linear_attention"),
        (1, "full_attention"),
        (2, "linear_attention"),
    ]

    class DerivedPool(HybridStateTensorPool):
        pass

    derived_pool = DerivedPool(pool.layout, 2, "cpu")
    _expect_error(
        lambda: _assemble(config=config, pool=derived_pool),
        ValueError,
        "exact HybridStateTensorPool",
    )


def test_assembly_rejects_bad_callback_and_preserves_pool():
    pool = _pool()
    before = _snapshot(pool)

    _expect_error(
        lambda: _assemble(
            pool=pool,
            callback=lambda *args: object(),
        ),
        ValueError,
        "Qwen35DecoderLayerShell",
    )
    _expect_error(
        lambda: _assemble(
            pool=pool,
            callback=lambda index, block, adapter: _decoder(
                "full_attention"
            ),
        ),
        ValueError,
        "block type",
    )

    def failing_callback(index, block, adapter):
        if index == 1:
            raise RuntimeError("injected assembly failure")
        return _decoder(block)

    _expect_error(
        lambda: _assemble(pool=pool, callback=failing_callback),
        RuntimeError,
        "injected assembly failure",
    )
    for key, tensor in before.items():
        torch.testing.assert_close(pool._tensors[key], tensor)


def test_assembled_model_integrates_with_owner_and_transactional_run():
    pool, _, assembly = _assemble()
    owner = build_qwen35_hybrid_model_owner(assembly.model)
    assert owner.model is assembly.model
    assert owner.layer_stack is assembly.layer_stack
    assert owner.state_transaction is assembly.state_transaction
    assert owner.pool is pool

    leases = (
        HybridStateLease(0, 1, 301),
        HybridStateLease(1, 1, 302),
    )
    for lease in leases:
        pool.activate(lease)
    before = _snapshot(pool)
    hidden, logits = assembly.model.run_step(
        leases,
        (1, 2),
        torch.tensor([1, 2, 3], dtype=torch.int64),
        torch.arange(3, dtype=torch.int64),
    )
    assert hidden.shape == (3, 3)
    assert logits.shape == (3, 2)
    assert any(
        not torch.equal(pool._tensors[key], tensor)
        for key, tensor in before.items()
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 root model assembly factory tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
