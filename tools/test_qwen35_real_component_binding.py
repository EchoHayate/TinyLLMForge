from __future__ import annotations

import builtins
import json
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
from tinyvllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tinyvllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedHeadPairedColumnParallelLinear,
    ReplicatedKVHeadParallelLinear,
    ReplicatedLocalOutputLinear,
    ReplicatedLinear,
    ReplicatedMergedColumnParallelLinear,
    ReplicatedWeightRowParallelLinear,
    RowParallelLinear,
    SegmentedColumnParallelLinear,
)
from tinyvllm.layers.qwen35_full_attention import Qwen35FullAttentionShell
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.models.qwen35_checkpoint import (
    build_qwen35_checkpoint_tensor_plan,
)
from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    prepare_qwen35_checkpoint_candidate_target,
)


CONFIG_PATH = Path("/tmp/qwen35-2b-15852e8-config.json")
INDEX_PATH = Path(
    "/tmp/qwen35-2b-15852e8-model.safetensors.index.json"
)
HEADER_PATH = Path("/tmp/qwen35-safetensors-header.json")


def _namespace(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{
            key: _namespace(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_namespace(item) for item in value)
    return value


class _StaticAttentionBackend(nn.Module):

    def __init__(
        self,
        layer_index,
        local_query_heads,
        local_kv_heads,
        head_dim,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.local_query_heads = local_query_heads
        self.local_kv_heads = local_kv_heads
        self.head_dim = head_dim

    def forward(self, *_):
        raise AssertionError("attention backend must not execute")


def _load_metadata():
    for path in (CONFIG_PATH, INDEX_PATH, HEADER_PATH):
        if not path.is_file():
            raise AssertionError(f"missing verified metadata file: {path}")
    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    index_payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    header_payload = json.loads(HEADER_PATH.read_text(encoding="utf-8"))
    shards = set(index_payload["weight_map"].values())
    assert len(shards) == 1
    shard_name = next(iter(shards))
    return (
        _namespace(config_payload),
        index_payload,
        {shard_name: header_payload},
    )


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


def _snapshot_registrations(model):
    return tuple(
        (
            name,
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )
        for name, tensor in (
            list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False))
        )
    )


def _install_execution_guards(model):
    def forbidden(*_, **__):
        raise AssertionError(
            "checkpoint binding must not execute loaders or model forward"
        )

    model.run_step = forbidden
    for tensor in model.parameters():
        if callable(getattr(tensor, "weight_loader", None)):
            tensor.weight_loader = forbidden


def _assert_exact_component_types(model):
    assert type(model.embed_tokens) is VocabParallelEmbedding
    assert type(model.lm_head) is ParallelLMHead
    assert type(model.final_norm) is Qwen35OffsetRMSNorm
    for layer in model.layer_stack.layers:
        assert type(layer.input_layernorm) is Qwen35OffsetRMSNorm
        assert type(layer.post_attention_layernorm) is Qwen35OffsetRMSNorm
        assert type(layer.mlp.gate_up_proj) is (
            ReplicatedMergedColumnParallelLinear
        )
        assert type(layer.mlp.down_proj) is ReplicatedLinear
        if layer.block_type == "linear_attention":
            component = layer.linear_attention
            assert type(component) is Qwen35LinearAttentionShell
            assert type(component.in_proj_qkv) is (
                SegmentedColumnParallelLinear
            )
            assert type(component.in_proj_z) is ColumnParallelLinear
            assert type(component.in_proj_b) is (
                ReplicatedLocalOutputLinear
            )
            assert type(component.in_proj_a) is (
                ReplicatedLocalOutputLinear
            )
            assert type(component.out_proj) is (
                ReplicatedWeightRowParallelLinear
            )
        else:
            component = layer.full_attention
            assert type(component) is Qwen35FullAttentionShell
            assert type(component.q_projection) is (
                ReplicatedHeadPairedColumnParallelLinear
            )
            assert type(component.k_projection) is (
                ReplicatedKVHeadParallelLinear
            )
            assert type(component.v_projection) is (
                ReplicatedKVHeadParallelLinear
            )
            assert type(component.output_projection) is RowParallelLinear
            assert type(component.q_norm) is Qwen35OffsetRMSNorm
            assert type(component.k_norm) is Qwen35OffsetRMSNorm


def test_real_24_layer_graph_binds_all_320_entries_read_only():
    config, index_payload, shard_headers = _load_metadata()
    tensor_plan = build_qwen35_checkpoint_tensor_plan(
        config,
        index_payload,
        shard_headers,
    )
    assert len(tensor_plan.loads) == 320
    assert len(tensor_plan.skips) == 312
    assert tensor_plan.payload_bytes == 4_548_144_832
    assert sum(
        load.metadata.dtype == "BF16"
        for load in tensor_plan.loads
    ) == 284
    assert sum(
        load.metadata.dtype == "F32"
        for load in tensor_plan.loads
    ) == 36
    assert sum(
        load.transform == "squeeze_conv_channel"
        for load in tensor_plan.loads
    ) == 18

    text_config = config.text_config
    assert text_config.num_hidden_layers == 24
    assert text_config.layer_types.count("linear_attention") == 18
    assert text_config.layer_types.count("full_attention") == 6

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            raise AssertionError("safetensors payload must not be opened")
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        for world_size in (1, 2):
            for rank in range(world_size):
                layout = build_qwen35_hybrid_state_layout(
                    config,
                    tensor_parallel_size=world_size,
                    dtype=torch.bfloat16,
                )
                pool = HybridStateTensorPool(
                    layout,
                    capacity=1,
                    device="cpu",
                )
                pool_snapshot = _snapshot_pool(pool)
                backend_calls = []

                def build_backend(
                    layer_index,
                    local_query_heads,
                    local_kv_heads,
                    head_dim,
                ):
                    arguments = (
                        layer_index,
                        local_query_heads,
                        local_kv_heads,
                        head_dim,
                    )
                    backend_calls.append(arguments)
                    return _StaticAttentionBackend(*arguments)

                target = prepare_qwen35_checkpoint_candidate_target(
                    config,
                    tensor_plan,
                    pool=pool,
                    tensor_parallel_size=world_size,
                    tensor_parallel_rank=rank,
                    build_attention_backend=build_backend,
                    parameter_device="meta",
                )
                assembly = target.assembly
                model = assembly.packed.model
                assert len(model.layer_stack.layers) == 24
                assert tuple(
                    layer.block_type
                    for layer in model.layer_stack.layers
                ) == text_config.layer_types
                assert len(assembly.packed.adapters) == 18
                assert len(backend_calls) == 6
                assert all(
                    call[1:] == (
                        text_config.num_attention_heads // world_size,
                        text_config.num_key_value_heads // world_size,
                        text_config.head_dim,
                    )
                    for call in backend_calls
                )
                assert model.embed_tokens.weight is model.lm_head.weight
                _assert_exact_component_types(model)
                assert all(
                    tensor.device.type == "meta"
                    for _, tensor in (
                        list(model.named_parameters(remove_duplicate=False))
                        + [
                            (name, tensor)
                            for name, tensor in model.named_buffers(
                                remove_duplicate=False
                            )
                            if not name.endswith("rotary.inv_freq")
                        ]
                    )
                )
                _install_execution_guards(model)
                registrations = _snapshot_registrations(model)

                candidate_model, binding_plan = target.take()

                assert candidate_model is model
                assert target.pool is pool
                assert target.binding_plan is binding_plan
                assert len(binding_plan.bindings) == 320
                assert sum(
                    binding.load.weight.target
                    in ("embed_tokens.weight", "final_norm.weight")
                    for binding in binding_plan.bindings
                ) == 2
                assert sum(
                    binding.load.weight.target.startswith("layers.")
                    and text_config.layer_types[
                        int(binding.load.weight.target.split(".")[1])
                    ] == "linear_attention"
                    for binding in binding_plan.bindings
                ) == 252
                assert sum(
                    binding.load.weight.target.startswith("layers.")
                    and text_config.layer_types[
                        int(binding.load.weight.target.split(".")[1])
                    ] == "full_attention"
                    for binding in binding_plan.bindings
                ) == 66
                assert sum(
                    binding.destination_kind == "buffer"
                    for binding in binding_plan.bindings
                ) == 72
                assert sum(
                    binding.destination.dtype == torch.float32
                    for binding in binding_plan.bindings
                ) == 18
                assert all(
                    binding.destination.device.type == "meta"
                    for binding in binding_plan.bindings
                )
                assert _snapshot_registrations(model) == registrations
                _assert_pool_unchanged(pool, pool_snapshot)
    finally:
        builtins.open = original_open


def main():
    test_real_24_layer_graph_binds_all_320_entries_read_only()
    print("qwen35 real component binding tests passed (1 test)")


if __name__ == "__main__":
    main()
