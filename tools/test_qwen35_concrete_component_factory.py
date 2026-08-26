from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
    "tinyvllm.utils",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.engine.qwen35_hybrid_state import (
    build_qwen35_hybrid_state_layout,
)
from tinyvllm.layers import linear as linear_module
from tinyvllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tinyvllm.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedColumnParallelLinear,
    ReplicatedHeadPairedColumnParallelLinear,
    ReplicatedKVHeadParallelLinear,
    ReplicatedSegmentedColumnParallelLinear,
    RowParallelLinear,
)
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_full_attention import Qwen35FullAttentionShell
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.layers.qwen35_rotary_embedding import (
    Qwen35PartialInterleavedRotaryEmbedding,
)
from tinyvllm.models.qwen35_components import (
    Qwen35ConcreteComponentAssembly,
    _Qwen35MLP,
    build_qwen35_concrete_component_assembly,
)
from tinyvllm.models import qwen35_components as components_module


COMPUTE_DTYPE = torch.bfloat16


def _config(**overrides):
    values = {
        "dtype": "bfloat16",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "layer_types": (
            "linear_attention",
            "full_attention",
        ),
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "rope_parameters": {
            "rope_theta": 1_000_000,
            "partial_rotary_factor": 0.75,
            "mrope_section": (1, 1, 1),
        },
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _pool(config, world_size):
    layout = build_qwen35_hybrid_state_layout(
        config,
        tensor_parallel_size=world_size,
        dtype=COMPUTE_DTYPE,
    )
    return HybridStateTensorPool(layout, capacity=2, device="cpu")


def _snapshot_pool(pool):
    return {
        key: (
            id(tensor),
            tensor.clone(),
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )
        for key, tensor in pool._tensors.items()
    }


def _assert_pool_unchanged(pool, snapshot):
    assert set(pool._tensors) == set(snapshot)
    for key, tensor in pool._tensors.items():
        object_id, value, shape, dtype, device = snapshot[key]
        assert id(tensor) == object_id
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == device
        torch.testing.assert_close(tensor, value)


class _Backend(nn.Module):

    def __init__(self, arguments):
        super().__init__()
        self.arguments = arguments


def _build(config, world_size, rank, *, pool=None, callback=None, **kwargs):
    pool = _pool(config, world_size) if pool is None else pool
    calls = []

    def default_callback(layer_index, local_query_heads, local_kv_heads, head_dim):
        arguments = (
            layer_index,
            local_query_heads,
            local_kv_heads,
            head_dim,
        )
        calls.append(arguments)
        return _Backend(arguments)

    assembly = build_qwen35_concrete_component_assembly(
        config,
        pool=pool,
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
        build_attention_backend=(
            default_callback if callback is None else callback
        ),
        **kwargs,
    )
    return pool, calls, assembly


def _checkpoint_tensors(model):
    return tuple(
        list(model.named_parameters(remove_duplicate=False))
        + [
            (name, tensor)
            for name, tensor in model.named_buffers(remove_duplicate=False)
            if not name.endswith("rotary.inv_freq")
        ]
    )


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_cached_prefill_mlp_down_projection_uses_full_context_rows():
    original_rank = torch.distributed.get_rank
    original_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    try:
        mlp = _Qwen35MLP(hidden_size=4, intermediate_size=6)
    finally:
        torch.distributed.get_rank = original_rank
        torch.distributed.get_world_size = original_world_size

    hidden_states = torch.arange(
        8,
        dtype=torch.bfloat16,
    ).reshape(2, 4)
    gate_up = torch.arange(
        24,
        dtype=torch.bfloat16,
    ).reshape(2, 12)
    mlp.gate_up_proj.forward = lambda _hidden_states: gate_up
    down_inputs = []

    def record_down_projection(value):
        down_inputs.append(value.clone())
        return value[:, :4]

    mlp.down_proj.forward = record_down_projection
    original_get_context = components_module.get_context
    components_module.get_context = lambda: types.SimpleNamespace(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen_k=5,
    )
    try:
        output = mlp(hidden_states)
    finally:
        components_module.get_context = original_get_context

    assert down_inputs[0].shape == (5, 6)
    torch.testing.assert_close(
        down_inputs[0][:-2],
        torch.zeros(3, 6, dtype=torch.bfloat16),
    )
    assert output.shape == (2, 4)


def test_tp_1_and_2_construct_exact_meta_graph_and_reuse_pool():
    config = _config()
    for world_size in (1, 2):
        for rank in range(world_size):
            pool = _pool(config, world_size)
            snapshot = _snapshot_pool(pool)
            original_rank = torch.distributed.get_rank
            original_world_size = torch.distributed.get_world_size

            _, calls, assembly = _build(
                config,
                world_size,
                rank,
                pool=pool,
            )

            assert type(assembly) is Qwen35ConcreteComponentAssembly
            assert assembly.packed.pool is pool
            assert assembly.tensor_parallel_size == world_size
            assert assembly.tensor_parallel_rank == rank
            assert assembly.parameter_device == torch.device("meta")
            assert assembly.compute_dtype == COMPUTE_DTYPE
            assert assembly.stable_dtype == torch.float32
            assert torch.distributed.get_rank is original_rank
            assert torch.distributed.get_world_size is original_world_size

            model = assembly.packed.model
            assert type(model.embed_tokens) is VocabParallelEmbedding
            assert type(model.lm_head) is ParallelLMHead
            assert model.lm_head.exact_full_vocab is True
            assert type(model.final_norm) is Qwen35OffsetRMSNorm
            assert model.embed_tokens.weight is model.lm_head.weight
            assert model.embed_tokens.weight.shape == (
                config.vocab_size // world_size,
                config.hidden_size,
            )
            assert len(model.layer_stack.layers) == 2
            assert tuple(
                layer.block_type for layer in model.layer_stack.layers
            ) == config.layer_types
            assert calls == [(1, 2 // world_size, 2 // world_size, 8)]
            assert tuple(
                adapter.layer_index for adapter in assembly.packed.adapters
            ) == (0,)
            assert assembly.packed.adapters[0].pool is pool

            linear_layer, full_layer = model.layer_stack.layers
            assert type(linear_layer) is Qwen35DecoderLayerShell
            assert type(full_layer) is Qwen35DecoderLayerShell
            for layer in (linear_layer, full_layer):
                assert type(layer.input_layernorm) is Qwen35OffsetRMSNorm
                assert type(layer.post_attention_layernorm) is (
                    Qwen35OffsetRMSNorm
                )
                assert type(layer.mlp.gate_up_proj) is (
                    linear_module.ReplicatedMergedColumnParallelLinear
                )
                assert type(layer.mlp.down_proj) is (
                    linear_module.ReplicatedLinear
                )
                assert layer.mlp.gate_up_proj.tp_size == world_size
                assert layer.mlp.gate_up_proj.tp_rank == rank

            linear = linear_layer.linear_attention
            assert type(linear) is Qwen35LinearAttentionShell
            assert type(linear.in_proj_qkv) is (
                ReplicatedSegmentedColumnParallelLinear
            )
            assert type(linear.in_proj_z) is ReplicatedColumnParallelLinear
            replicated_local_output = getattr(
                linear_module,
                "ReplicatedLocalOutputLinear",
                None,
            )
            assert replicated_local_output is not None
            assert type(linear.in_proj_b) is replicated_local_output
            assert type(linear.in_proj_a) is replicated_local_output
            assert linear.in_proj_b.weight.shape == (
                config.linear_num_value_heads,
                config.hidden_size,
            )
            assert linear.in_proj_a.weight.shape == (
                config.linear_num_value_heads,
                config.hidden_size,
            )
            assert type(linear.out_proj) is RowParallelLinear
            assert (
                linear.out_proj.accumulation_dtype
                is torch.float32
            )
            assert linear.out_proj.preserve_dense_prefill is True
            assert linear.out_proj.weight.shape == (
                config.hidden_size,
                (
                    config.linear_num_value_heads
                    * config.linear_value_head_dim
                    // world_size
                ),
            )
            assert linear.local_key_heads == 2 // world_size
            assert linear.local_value_heads == 2 // world_size
            assert linear.conv_weight.shape == (12 // world_size, 3)
            assert linear.A_log.shape == (2 // world_size,)
            assert linear.dt_bias.shape == (2 // world_size,)
            assert linear.norm_weight.shape == (2,)
            assert linear.conv_weight.dtype == COMPUTE_DTYPE
            assert linear.dt_bias.dtype == COMPUTE_DTYPE
            assert linear.A_log.dtype == torch.float32
            assert linear.norm_weight.dtype == torch.bfloat16

            full = full_layer.full_attention
            assert type(full) is Qwen35FullAttentionShell
            assert type(full.q_projection) is (
                ReplicatedHeadPairedColumnParallelLinear
            )
            assert type(full.k_projection) is (
                ReplicatedKVHeadParallelLinear
            )
            assert type(full.v_projection) is (
                ReplicatedKVHeadParallelLinear
            )
            assert full.q_projection.weight.shape == (
                config.num_attention_heads * 2 * config.head_dim,
                config.hidden_size,
            )
            for projection in (
                full.k_projection,
                full.v_projection,
            ):
                assert projection.local_num_kv_heads == (
                    2 // world_size
                )
                assert projection.num_kv_head_replicas == 1
                assert projection.source_kv_rank == rank
                assert projection.weight.shape == (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                )
                assert (
                    projection.requires_unpartitioned_linear_execution
                    is True
                )
            assert (
                full.q_projection.requires_unpartitioned_linear_execution
                is True
            )
            assert type(full.output_projection) is RowParallelLinear
            assert (
                full.output_projection.accumulation_dtype
                is torch.float32
            )
            assert (
                full.output_projection.preserve_dense_prefill
                is True
            )
            assert full.output_projection.weight.shape == (
                config.hidden_size,
                (
                    config.num_attention_heads
                    * config.head_dim
                    // world_size
                ),
            )
            assert type(full.q_norm) is Qwen35OffsetRMSNorm
            assert type(full.k_norm) is Qwen35OffsetRMSNorm
            assert type(full.rotary) is (
                Qwen35PartialInterleavedRotaryEmbedding
            )
            assert type(full.attention_backend) is _Backend
            assert full.local_query_heads == 2 // world_size
            assert full.local_kv_heads == 2 // world_size
            assert full.rotary.rotary_dim == 6
            assert full.rotary.mrope_section == (1, 1, 1)

            checkpoint_tensors = _checkpoint_tensors(model)
            assert checkpoint_tensors
            assert all(
                tensor.device.type == "meta"
                for _, tensor in checkpoint_tensors
            )
            assert all(
                tensor.dtype == (
                    torch.float32
                    if name.endswith("linear_attention.A_log")
                    else COMPUTE_DTYPE
                )
                for name, tensor in checkpoint_tensors
            )
            _assert_pool_unchanged(pool, snapshot)


def test_tp4_replicates_complete_full_attention_kv_heads():
    config = _config(
        hidden_size=16,
        intermediate_size=16,
        vocab_size=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
    )
    observed = []
    for rank in range(4):
        _, calls, assembly = _build(config, 4, rank)
        full = assembly.packed.model.layer_stack.layers[
            1
        ].full_attention
        assert calls == [(1, 2, 1, 8)]
        assert full.local_query_heads == 2
        assert full.local_kv_heads == 1
        assert type(full.q_projection) is (
            ReplicatedHeadPairedColumnParallelLinear
        )
        assert type(full.k_projection) is (
            ReplicatedKVHeadParallelLinear
        )
        assert type(full.v_projection) is (
            ReplicatedKVHeadParallelLinear
        )
        assert full.q_projection.weight.shape == (128, 16)
        for projection in (
            full.k_projection,
            full.v_projection,
        ):
            assert projection.local_num_kv_heads == 1
            assert projection.num_kv_head_replicas == 2
            assert projection.source_kv_rank == rank // 2
            assert projection.weight.shape == (16, 16)
        observed.append((
            full.k_projection.source_kv_rank,
            full.v_projection.source_kv_rank,
        ))
    assert observed == [(0, 0), (0, 0), (1, 1), (1, 1)]


def test_untied_output_head_owns_distinct_sharded_weight():
    config = _config(tie_word_embeddings=False)

    for world_size in (1, 2):
        for rank in range(world_size):
            _, _, assembly = _build(config, world_size, rank)
            model = assembly.packed.model

            assert model.embed_tokens.weight is not model.lm_head.weight
            assert model.embed_tokens.weight.shape == (
                config.vocab_size // world_size,
                config.hidden_size,
            )
            assert model.lm_head.weight.shape == (
                config.vocab_size // world_size,
                config.hidden_size,
            )
            assert callable(model.lm_head.weight.weight_loader)


def test_factory_failures_restore_tp_context_and_preserve_pool():
    config = _config()
    original_rank = torch.distributed.get_rank
    original_world_size = torch.distributed.get_world_size

    cases = (
        (_config(dtype="float16"), 1, 0, {}, "dtype"),
        (_config(hidden_act="gelu"), 1, 0, {}, "hidden_act"),
        (
            _config(tie_word_embeddings="false"),
            1,
            0,
            {},
            "tie_word_embeddings",
        ),
        (_config(vocab_size=31), 2, 0, {}, "divisible"),
        (_config(rope_parameters={}), 1, 0, {}, "rope_theta"),
        (
            _config(rope_parameters={
                "rope_theta": 1_000_000,
                "partial_rotary_factor": 0.75,
                "mrope_section": (1, 1, 2),
            }),
            1,
            0,
            {},
            "mrope_section",
        ),
        (
            config,
            1,
            0,
            {"parameter_device": "cuda"},
            "parameter_device",
        ),
    )
    for case_config, world_size, rank, kwargs, message in cases:
        pool = _pool(config, world_size)
        snapshot = _snapshot_pool(pool)
        _expect_error(
            lambda case_config=case_config,
            world_size=world_size,
            rank=rank,
            kwargs=kwargs,
            pool=pool: _build(
                case_config,
                world_size,
                rank,
                pool=pool,
                **kwargs,
            ),
            message,
        )
        assert torch.distributed.get_rank is original_rank
        assert torch.distributed.get_world_size is original_world_size
        _assert_pool_unchanged(pool, snapshot)

    pool = _pool(config, 1)
    for world_size, rank, message in (
        (True, 0, "tensor_parallel_size"),
        (0, 0, "tensor_parallel_size"),
        (1, -1, "tensor_parallel_rank"),
        (1, 1, "tensor_parallel_rank"),
    ):
        _expect_error(
            lambda world_size=world_size, rank=rank: _build(
                config,
                world_size,
                rank,
                pool=pool,
            ),
            message,
        )

    _expect_error(
        lambda: build_qwen35_concrete_component_assembly(
            config,
            pool=pool,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            build_attention_backend=None,
        ),
        "build_attention_backend",
    )

    def bad_callback(*_):
        return object()

    _expect_error(
        lambda: _build(
            config,
            1,
            0,
            pool=pool,
            callback=bad_callback,
        ),
        "attention backend",
    )
    mismatch_pool = _pool(
        _config(layer_types=("full_attention", "linear_attention")),
        1,
    )
    _expect_error(
        lambda: _build(config, 1, 0, pool=mismatch_pool),
        "pool linear layer indices",
    )

    def raising_callback(*_):
        raise ValueError("backend failure")

    _expect_error(
        lambda: _build(
            config,
            1,
            0,
            pool=pool,
            callback=raising_callback,
        ),
        "backend failure",
    )
    assert torch.distributed.get_rank is original_rank
    assert torch.distributed.get_world_size is original_world_size


def main():
    test_tp_1_and_2_construct_exact_meta_graph_and_reuse_pool()
    test_tp4_replicates_complete_full_attention_kv_heads()
    test_untied_output_head_owns_distinct_sharded_weight()
    test_factory_failures_restore_tp_context_and_preserve_pool()
    print("qwen35 concrete component factory tests passed (2 tests)")


if __name__ == "__main__":
    main()
