from __future__ import annotations

from contextlib import contextmanager
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

from tinyvllm.engine.hybrid_state import (
    HybridStateComponentSpec,
    HybridStateLayout,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.layers import linear as linear_module
from tinyvllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tinyvllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedColumnParallelLinear,
    ReplicatedHeadPairedColumnParallelLinear,
    ReplicatedKVHeadParallelLinear,
    ReplicatedLinear,
    ReplicatedMergedColumnParallelLinear,
    ReplicatedSegmentedColumnParallelLinear,
    RowParallelLinear,
)
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_full_attention import Qwen35FullAttentionShell
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)
from tinyvllm.layers.qwen35_packed_layer_stack import (
    Qwen35PackedHeterogeneousLayerStack,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointLoadTarget,
    Qwen35CheckpointSource,
    Qwen35CheckpointTensorLoad,
    Qwen35CheckpointTensorMetadata,
    Qwen35CheckpointTensorPlan,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    build_qwen35_checkpoint_binding_plan,
)
from tinyvllm.models.qwen35_components import (
    build_qwen35_concrete_component_assembly,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 12
VOCAB_SIZE = 32
KEY_HEADS = 2
VALUE_HEADS = 2
KEY_HEAD_DIM = 2
VALUE_HEAD_DIM = 2
FULL_QUERY_HEADS = 2
FULL_KV_HEADS = 2
FULL_HEAD_DIM = 2
CONV_KERNEL = 3
COMPUTE_DTYPE = torch.bfloat16


@contextmanager
def _dist_layout(rank: int, world_size: int):
    old_rank = torch.distributed.get_rank
    old_world_size = torch.distributed.get_world_size
    old_all_reduce = torch.distributed.all_reduce
    old_gather = torch.distributed.gather
    torch.distributed.get_rank = lambda: rank
    torch.distributed.get_world_size = lambda: world_size
    torch.distributed.all_reduce = lambda tensor: tensor
    torch.distributed.gather = lambda tensor, outputs, root: None
    try:
        yield
    finally:
        torch.distributed.get_rank = old_rank
        torch.distributed.get_world_size = old_world_size
        torch.distributed.all_reduce = old_all_reduce
        torch.distributed.gather = old_gather


class _MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_up_proj = ReplicatedMergedColumnParallelLinear(
            HIDDEN_SIZE,
            [INTERMEDIATE_SIZE, INTERMEDIATE_SIZE],
            bias=False,
        )
        self.down_proj = ReplicatedLinear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            bias=False,
        )

    def forward(self, tensor):
        return tensor


class _IdentityRotary(nn.Module):

    def forward(self, position_ids, query, key):
        return query, key


class _AttentionBackend(nn.Module):

    def forward(self, query, key, value):
        return query


def _bf16(module: nn.Module) -> nn.Module:
    return module.to(dtype=COMPUTE_DTYPE)


def _replicated_local_output_linear(
    input_size: int,
    output_size: int,
) -> nn.Module:
    projection_type = getattr(
        linear_module,
        "ReplicatedLocalOutputLinear",
        ColumnParallelLinear,
    )
    return projection_type(input_size, output_size, bias=False)


def _linear_attention(world_size: int) -> Qwen35LinearAttentionShell:
    local_key_heads = KEY_HEADS // world_size
    local_value_heads = VALUE_HEADS // world_size
    global_key_width = KEY_HEADS * KEY_HEAD_DIM
    global_value_width = VALUE_HEADS * VALUE_HEAD_DIM
    global_conv_width = 2 * global_key_width + global_value_width
    local_conv_width = global_conv_width // world_size
    return Qwen35LinearAttentionShell(
        local_key_heads=local_key_heads,
        local_value_heads=local_value_heads,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        norm_eps=1e-6,
        in_proj_qkv=_bf16(ReplicatedSegmentedColumnParallelLinear(
            HIDDEN_SIZE,
            (global_key_width, global_key_width, global_value_width),
            bias=False,
        )),
        in_proj_z=_bf16(ReplicatedColumnParallelLinear(
            HIDDEN_SIZE,
            global_value_width,
            bias=False,
        )),
        in_proj_b=_bf16(_replicated_local_output_linear(
            HIDDEN_SIZE,
            VALUE_HEADS,
        )),
        in_proj_a=_bf16(_replicated_local_output_linear(
            HIDDEN_SIZE,
            VALUE_HEADS,
        )),
        out_proj=_bf16(RowParallelLinear(
            global_value_width,
            HIDDEN_SIZE,
            bias=False,
        )),
        conv_weight=torch.zeros(
            local_conv_width,
            CONV_KERNEL,
            dtype=COMPUTE_DTYPE,
        ),
        A_log=torch.zeros(local_value_heads, dtype=torch.float32),
        dt_bias=torch.zeros(local_value_heads, dtype=COMPUTE_DTYPE),
        norm_weight=torch.zeros(VALUE_HEAD_DIM, dtype=torch.float32),
    )


def _full_attention(world_size: int) -> Qwen35FullAttentionShell:
    return Qwen35FullAttentionShell(
        head_dim=FULL_HEAD_DIM,
        local_query_heads=FULL_QUERY_HEADS // world_size,
        local_kv_heads=FULL_KV_HEADS // world_size,
        q_projection=_bf16(ReplicatedHeadPairedColumnParallelLinear(
            HIDDEN_SIZE,
            FULL_QUERY_HEADS,
            FULL_HEAD_DIM,
            bias=False,
        )),
        k_projection=_bf16(ReplicatedKVHeadParallelLinear(
            HIDDEN_SIZE,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            bias=False,
        )),
        v_projection=_bf16(ReplicatedKVHeadParallelLinear(
            HIDDEN_SIZE,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            bias=False,
        )),
        q_norm=_bf16(Qwen35OffsetRMSNorm(FULL_HEAD_DIM)),
        k_norm=_bf16(Qwen35OffsetRMSNorm(FULL_HEAD_DIM)),
        rotary=_IdentityRotary(),
        attention_backend=_AttentionBackend(),
        output_projection=_bf16(RowParallelLinear(
            FULL_QUERY_HEADS * FULL_HEAD_DIM,
            HIDDEN_SIZE,
            bias=False,
        )),
    )


def _fixture(
    rank: int,
    world_size: int,
    *,
    tie_word_embeddings: bool = True,
):
    with _dist_layout(rank, world_size):
        embed_tokens = _bf16(VocabParallelEmbedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
        ))
        lm_head = _bf16(ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE))
        if tie_word_embeddings:
            lm_head.weight.data = embed_tokens.weight.data
        linear_layer = Qwen35DecoderLayerShell(
            block_type="linear_attention",
            input_layernorm=_bf16(Qwen35OffsetRMSNorm(HIDDEN_SIZE)),
            post_attention_layernorm=_bf16(
                Qwen35OffsetRMSNorm(HIDDEN_SIZE)
            ),
            mlp=_bf16(_MLP()),
            linear_attention=_linear_attention(world_size),
        )
        full_layer = Qwen35DecoderLayerShell(
            block_type="full_attention",
            input_layernorm=_bf16(Qwen35OffsetRMSNorm(HIDDEN_SIZE)),
            post_attention_layernorm=_bf16(
                Qwen35OffsetRMSNorm(HIDDEN_SIZE)
            ),
            mlp=_bf16(_MLP()),
            full_attention=_full_attention(world_size),
        )
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (
                (
                    2 * KEY_HEADS * KEY_HEAD_DIM
                    + VALUE_HEADS * VALUE_HEAD_DIM
                ) // world_size,
                CONV_KERNEL,
            ),
            COMPUTE_DTYPE,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (
                VALUE_HEADS // world_size,
                VALUE_HEAD_DIM,
                KEY_HEAD_DIM,
            ),
            COMPUTE_DTYPE,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=2, device="cpu")
    adapter = Qwen35LayerStateAdapter(pool, 0)
    transaction = Qwen35CrossLayerStateTransaction((adapter,))
    stack = Qwen35PackedHeterogeneousLayerStack(
        (linear_layer, full_layer),
        transaction,
    )
    model = Qwen35PackedForCausalLM(
        embed_tokens,
        stack,
        _bf16(Qwen35OffsetRMSNorm(HIDDEN_SIZE)),
        lm_head,
    )
    return model, pool


def _tp4_replication_config():
    return types.SimpleNamespace(
        dtype="bfloat16",
        hidden_size=16,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        rope_parameters={
            "rope_theta": 1_000_000,
            "partial_rotary_factor": 0.75,
            "mrope_section": (1, 1, 1),
        },
    )


def _tp4_replication_fixture(rank: int):
    config = _tp4_replication_config()
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (6, 3),
            COMPUTE_DTYPE,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (1, 2, 2),
            COMPUTE_DTYPE,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=2, device="cpu")
    assembly = build_qwen35_concrete_component_assembly(
        config,
        pool=pool,
        tensor_parallel_size=4,
        tensor_parallel_rank=rank,
        build_attention_backend=(
            lambda *_: _AttentionBackend()
        ),
        parameter_device="cpu",
    )
    return assembly.packed.model, pool


def _tp4_replication_tensor_plan():
    entries = (
        "layers.1.full_attention.k_projection.weight",
        "layers.1.full_attention.v_projection.weight",
    )
    loads = []
    offset = 0
    for index, target in enumerate(entries):
        shape = (16, 16)
        payload_bytes = shape[0] * shape[1] * 2
        loads.append(Qwen35CheckpointTensorLoad(
            weight=Qwen35CheckpointLoadTarget(
                source=Qwen35CheckpointSource(
                    name=f"source.tp4.{index}.{target}",
                    shard="model.safetensors",
                ),
                target=target,
                packed_slot=None,
            ),
            metadata=Qwen35CheckpointTensorMetadata(
                dtype="BF16",
                shape=shape,
                data_offsets=(offset, offset + payload_bytes),
            ),
            transform="identity",
        ))
        offset += payload_bytes
    return Qwen35CheckpointTensorPlan(
        loads=tuple(loads),
        skips=(),
        payload_bytes=offset,
    )


def _tensor_plan(
    *,
    tie_word_embeddings: bool = True,
) -> Qwen35CheckpointTensorPlan:
    global_key_width = KEY_HEADS * KEY_HEAD_DIM
    global_value_width = VALUE_HEADS * VALUE_HEAD_DIM
    global_conv_width = 2 * global_key_width + global_value_width
    entries = [
        ("embed_tokens.weight", (VOCAB_SIZE, HIDDEN_SIZE), "BF16", None),
        ("final_norm.weight", (HIDDEN_SIZE,), "BF16", None),
    ]
    if not tie_word_embeddings:
        entries.append(
            ("lm_head.weight", (VOCAB_SIZE, HIDDEN_SIZE), "BF16", None)
        )
    for layer_index in (0, 1):
        prefix = f"layers.{layer_index}."
        entries.extend((
            (prefix + "input_layernorm.weight", (HIDDEN_SIZE,), "BF16", None),
            (
                prefix + "post_attention_layernorm.weight",
                (HIDDEN_SIZE,),
                "BF16",
                None,
            ),
            (
                prefix + "mlp.gate_up_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                "BF16",
                0,
            ),
            (
                prefix + "mlp.gate_up_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                "BF16",
                1,
            ),
            (
                prefix + "mlp.down_proj.weight",
                (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                "BF16",
                None,
            ),
        ))
    entries.extend((
        (
            "layers.0.linear_attention.in_proj_qkv.weight",
            (global_conv_width, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.in_proj_z.weight",
            (global_value_width, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.in_proj_b.weight",
            (VALUE_HEADS, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.in_proj_a.weight",
            (VALUE_HEADS, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.out_proj.weight",
            (HIDDEN_SIZE, global_value_width),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.conv_weight",
            (global_conv_width, 1, CONV_KERNEL),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.A_log",
            (VALUE_HEADS,),
            "F32",
            None,
        ),
        (
            "layers.0.linear_attention.dt_bias",
            (VALUE_HEADS,),
            "BF16",
            None,
        ),
        (
            "layers.0.linear_attention.norm_weight",
            (VALUE_HEAD_DIM,),
            "F32",
            None,
        ),
        (
            "layers.1.full_attention.q_projection.weight",
            (FULL_QUERY_HEADS * 2 * FULL_HEAD_DIM, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.1.full_attention.k_projection.weight",
            (FULL_KV_HEADS * FULL_HEAD_DIM, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.1.full_attention.v_projection.weight",
            (FULL_KV_HEADS * FULL_HEAD_DIM, HIDDEN_SIZE),
            "BF16",
            None,
        ),
        (
            "layers.1.full_attention.output_projection.weight",
            (HIDDEN_SIZE, FULL_QUERY_HEADS * FULL_HEAD_DIM),
            "BF16",
            None,
        ),
        (
            "layers.1.full_attention.q_norm.weight",
            (FULL_HEAD_DIM,),
            "BF16",
            None,
        ),
        (
            "layers.1.full_attention.k_norm.weight",
            (FULL_HEAD_DIM,),
            "BF16",
            None,
        ),
    ))
    loads = []
    offset = 0
    for index, (target, shape, dtype, packed_slot) in enumerate(entries):
        transform = (
            "squeeze_conv_channel"
            if target.endswith("linear_attention.conv_weight")
            else "identity"
        )
        byte_width = 2 if dtype == "BF16" else 4
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        end = offset + element_count * byte_width
        source = Qwen35CheckpointSource(
            name=f"source.{index:02d}.{target}",
            shard="model.safetensors",
        )
        loads.append(Qwen35CheckpointTensorLoad(
            weight=Qwen35CheckpointLoadTarget(
                source=source,
                target=target,
                packed_slot=packed_slot,
            ),
            metadata=Qwen35CheckpointTensorMetadata(
                dtype=dtype,
                shape=shape,
                data_offsets=(offset, end),
            ),
            transform=transform,
        ))
        offset = end
    return Qwen35CheckpointTensorPlan(
        loads=tuple(loads),
        skips=(),
        payload_bytes=offset,
    )


def _snapshot(model, pool):
    tensors = {}
    for name, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        tensors[name] = (
            id(tensor),
            tensor.detach().clone(),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
        )
    pool_values = {
        key: tensor.clone()
        for key, tensor in pool._tensors.items()
    }
    return tensors, pool_values


def _assert_unchanged(model, pool, snapshot):
    expected_tensors, expected_pool = snapshot
    actual = dict(
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    assert set(actual) == set(expected_tensors)
    for name, tensor in actual.items():
        (
            object_id,
            value,
            pointer,
            storage_offset,
            dtype,
            device,
        ) = expected_tensors[name]
        assert id(tensor) == object_id
        assert tensor.untyped_storage().data_ptr() == pointer
        assert tensor.storage_offset() == storage_offset
        assert tensor.dtype == dtype
        assert tensor.device == device
        torch.testing.assert_close(tensor, value, equal_nan=True)
    for key, value in expected_pool.items():
        torch.testing.assert_close(pool._tensors[key], value)


def _binding_by_target(plan, target, packed_slot=None):
    matches = [
        binding
        for binding in plan.bindings
        if binding.load.weight.target == target
        and binding.load.weight.packed_slot == packed_slot
    ]
    assert len(matches) == 1
    return matches[0]


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_tp_1_and_2_bind_complete_real_component_graph_read_only():
    tensor_plan = _tensor_plan()
    assert len(tensor_plan.loads) == 27
    for world_size in (1, 2):
        for rank in range(world_size):
            model, pool = _fixture(rank, world_size)
            snapshot = _snapshot(model, pool)
            plan = build_qwen35_checkpoint_binding_plan(
                model,
                tensor_plan,
                tensor_parallel_size=world_size,
                tensor_parallel_rank=rank,
            )
            assert len(plan.bindings) == 27
            assert plan.tensor_parallel_size == world_size
            assert plan.tensor_parallel_rank == rank

            embedding = _binding_by_target(
                plan,
                "embed_tokens.weight",
            )
            assert embedding.destination_kind == "parameter"
            assert embedding.loader_kind == "custom_parameter_loader"
            assert embedding.local_shape == (
                VOCAB_SIZE // world_size,
                HIDDEN_SIZE,
            )

            gate = _binding_by_target(
                plan,
                "layers.0.mlp.gate_up_proj.weight",
                0,
            )
            up = _binding_by_target(
                plan,
                "layers.0.mlp.gate_up_proj.weight",
                1,
            )
            assert gate.destination is up.destination
            assert gate.local_shape == (INTERMEDIATE_SIZE, HIDDEN_SIZE)
            assert gate.destination_slice == (0, INTERMEDIATE_SIZE)
            assert up.destination_slice == (
                INTERMEDIATE_SIZE,
                INTERMEDIATE_SIZE,
            )
            down = _binding_by_target(
                plan,
                "layers.0.mlp.down_proj.weight",
            )
            assert down.local_shape == (
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
            )
            assert down.destination_slice is None

            conv = _binding_by_target(
                plan,
                "layers.0.linear_attention.conv_weight",
            )
            assert conv.destination_kind == "buffer"
            assert conv.loader_kind == "direct_buffer_copy"
            assert conv.local_shape == (
                (
                    2 * KEY_HEADS * KEY_HEAD_DIM
                    + VALUE_HEADS * VALUE_HEAD_DIM
                ) // world_size,
                CONV_KERNEL,
            )
            stable = _binding_by_target(
                plan,
                "layers.0.linear_attention.A_log",
            )
            assert stable.destination.dtype == torch.float32
            compute = _binding_by_target(
                plan,
                "layers.0.linear_attention.dt_bias",
            )
            assert compute.destination.dtype == COMPUTE_DTYPE
            qkv = _binding_by_target(
                plan,
                "layers.0.linear_attention.in_proj_qkv.weight",
            )
            assert type(
                qkv.destination.weight_loader.__self__
            ) is ReplicatedSegmentedColumnParallelLinear
            assert qkv.local_shape == (
                2 * KEY_HEADS * KEY_HEAD_DIM
                + VALUE_HEADS * VALUE_HEAD_DIM,
                HIDDEN_SIZE,
            )
            assert qkv.destination.shape == qkv.local_shape
            assert qkv.destination_slice is None
            z = _binding_by_target(
                plan,
                "layers.0.linear_attention.in_proj_z.weight",
            )
            assert type(
                z.destination.weight_loader.__self__
            ) is ReplicatedColumnParallelLinear
            assert z.local_shape == (
                VALUE_HEADS * VALUE_HEAD_DIM,
                HIDDEN_SIZE,
            )
            assert z.destination.shape == z.local_shape
            assert z.destination_slice is None
            replicated_local_output = getattr(
                linear_module,
                "ReplicatedLocalOutputLinear",
                None,
            )
            assert replicated_local_output is not None
            for suffix in ("in_proj_b.weight", "in_proj_a.weight"):
                projection = _binding_by_target(
                    plan,
                    f"layers.0.linear_attention.{suffix}",
                )
                owner = projection.destination.weight_loader.__self__
                assert type(owner) is replicated_local_output
                assert projection.local_shape == (
                    VALUE_HEADS,
                    HIDDEN_SIZE,
                )
                assert projection.destination.shape == (
                    VALUE_HEADS,
                    HIDDEN_SIZE,
                )
                assert (
                    projection.loader_kind
                    == "custom_parameter_loader"
                )

            assert (
                model.embed_tokens.weight.untyped_storage().data_ptr()
                == model.lm_head.weight.untyped_storage().data_ptr()
            )
            assert (
                model.embed_tokens.weight.storage_offset()
                == model.lm_head.weight.storage_offset()
            )
            _assert_unchanged(model, pool, snapshot)


def test_tp4_binds_complete_replicated_kv_heads_by_source_rank():
    tensor_plan = _tp4_replication_tensor_plan()
    observed = []
    for rank in range(4):
        model, pool = _tp4_replication_fixture(rank)
        snapshot = _snapshot(model, pool)
        plan = build_qwen35_checkpoint_binding_plan(
            model,
            tensor_plan,
            tensor_parallel_size=4,
            tensor_parallel_rank=rank,
        )
        assert len(plan.bindings) == 2
        for suffix in ("k_projection.weight", "v_projection.weight"):
            binding = _binding_by_target(
                plan,
                f"layers.1.full_attention.{suffix}",
            )
            projection = binding.destination.weight_loader.__self__
            assert type(projection) is ReplicatedKVHeadParallelLinear
            assert binding.local_shape == (16, 16)
            assert binding.destination.shape == (16, 16)
            assert binding.loader_kind == "custom_parameter_loader"
            assert projection.source_kv_rank == rank // 2
            observed.append((
                rank,
                suffix,
                projection.source_kv_rank,
            ))
        _assert_unchanged(model, pool, snapshot)
    assert tuple(row[2] for row in observed) == (
        0, 0,
        0, 0,
        1, 1,
        1, 1,
    )


def test_fail_closed_component_shape_dtype_loader_alias_and_plan_contracts():
    tensor_plan = _tensor_plan()

    _expect_error(
        lambda: build_qwen35_checkpoint_binding_plan(
            nn.Module(),
            tensor_plan,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        ),
        "exact Qwen35PackedForCausalLM",
    )
    model, _ = _fixture(0, 1)
    for size, rank, message in (
        (True, 0, "tensor_parallel_size"),
        (0, 0, "tensor_parallel_size"),
        (1, -1, "tensor_parallel_rank"),
        (1, 1, "tensor_parallel_rank"),
    ):
        _expect_error(
            lambda size=size, rank=rank: (
                build_qwen35_checkpoint_binding_plan(
                    model,
                    tensor_plan,
                    tensor_parallel_size=size,
                    tensor_parallel_rank=rank,
                )
            ),
            message,
        )

    cases = []

    model, pool = _fixture(0, 1)
    model.layer_stack.layers[0].block_type = "full_attention"
    cases.append((model, pool, tensor_plan, "block type"))

    model, pool = _fixture(0, 1)
    with _dist_layout(0, 1):
        model.layer_stack.layers[1].full_attention.q_projection = _bf16(
            ColumnParallelLinear(
                HIDDEN_SIZE,
                FULL_QUERY_HEADS * 2 * FULL_HEAD_DIM,
                bias=False,
            )
        )
    cases.append((
        model,
        pool,
        tensor_plan,
        "ReplicatedHeadPairedColumnParallelLinear",
    ))

    model, pool = _fixture(0, 1)
    model.layer_stack.layers[0].linear_attention.conv_weight = nn.Parameter(
        model.layer_stack.layers[0].linear_attention.conv_weight.clone()
    )
    cases.append((model, pool, tensor_plan, "registered buffer"))

    model, pool = _fixture(0, 1)
    model.layer_stack.layers[0].linear_attention.A_log = torch.zeros(
        VALUE_HEADS + 1,
        dtype=torch.float32,
    )
    cases.append((model, pool, tensor_plan, "local shape"))

    model, pool = _fixture(0, 1)
    model.layer_stack.layers[0].linear_attention.A_log = (
        model.layer_stack.layers[0].linear_attention.A_log.to(
            COMPUTE_DTYPE
        )
    )
    cases.append((model, pool, tensor_plan, "dtype"))

    model, pool = _fixture(0, 1)
    delattr(model.embed_tokens.weight, "weight_loader")
    cases.append((model, pool, tensor_plan, "weight_loader"))

    model, pool = _fixture(0, 1)
    model.embed_tokens.tp_rank = 1
    cases.append((model, pool, tensor_plan, "tp_rank"))

    model, pool = _fixture(0, 1)
    model.lm_head.weight = nn.Parameter(
        model.embed_tokens.weight.detach().clone()
    )
    cases.append((model, pool, tensor_plan, "share storage"))

    bad_slot_load = replace(
        tensor_plan.loads[4],
        weight=replace(
            tensor_plan.loads[4].weight,
            packed_slot=2,
        ),
    )
    cases.append((
        *_fixture(0, 1),
        replace(
            tensor_plan,
            loads=(
                *tensor_plan.loads[:4],
                bad_slot_load,
                *tensor_plan.loads[5:],
            ),
        ),
        "packed slot",
    ))

    missing_target_load = replace(
        tensor_plan.loads[0],
        weight=replace(
            tensor_plan.loads[0].weight,
            target="embed_tokens.missing",
        ),
    )
    cases.append((
        *_fixture(0, 1),
        replace(
            tensor_plan,
            loads=(
                missing_target_load,
                *tensor_plan.loads[1:],
            ),
        ),
        "missing",
    ))

    cases.append((
        *_fixture(0, 1),
        replace(
            tensor_plan,
            loads=(
                *tensor_plan.loads,
                tensor_plan.loads[0],
            ),
        ),
        "duplicate",
    ))

    for case_model, case_pool, case_plan, message in cases:
        snapshot = _snapshot(case_model, case_pool)
        _expect_error(
            lambda case_model=case_model, case_plan=case_plan: (
                build_qwen35_checkpoint_binding_plan(
                    case_model,
                    case_plan,
                    tensor_parallel_size=1,
                    tensor_parallel_rank=0,
                )
            ),
            message,
        )
        _assert_unchanged(case_model, case_pool, snapshot)


def test_meta_embedding_alias_requires_the_same_parameter_object():
    tensor_plan = _tensor_plan()
    model, _ = _fixture(0, 1)
    shared = nn.Parameter(torch.empty(
        model.embed_tokens.weight.shape,
        dtype=model.embed_tokens.weight.dtype,
        device="meta",
    ))
    shared.weight_loader = model.embed_tokens.weight.weight_loader
    model.embed_tokens.weight = shared
    model.lm_head.weight = shared

    plan = build_qwen35_checkpoint_binding_plan(
        model,
        tensor_plan,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    assert len(plan.bindings) == 27
    assert model.embed_tokens.weight is model.lm_head.weight

    independent = nn.Parameter(torch.empty(
        shared.shape,
        dtype=shared.dtype,
        device="meta",
    ))
    independent.weight_loader = model.lm_head.weight.weight_loader
    model.lm_head.weight = independent
    assert shared.untyped_storage().data_ptr() == 0
    assert independent.untyped_storage().data_ptr() == 0
    assert shared is not independent
    _expect_error(
        lambda: build_qwen35_checkpoint_binding_plan(
            model,
            tensor_plan,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        ),
        "embed_tokens and lm_head must share storage",
    )


def test_untied_lm_head_binds_independent_tp_shard():
    tensor_plan = _tensor_plan(tie_word_embeddings=False)
    for world_size in (1, 2):
        for rank in range(world_size):
            model, pool = _fixture(
                rank,
                world_size,
                tie_word_embeddings=False,
            )
            snapshot = _snapshot(model, pool)

            plan = build_qwen35_checkpoint_binding_plan(
                model,
                tensor_plan,
                tensor_parallel_size=world_size,
                tensor_parallel_rank=rank,
            )

            embedding = _binding_by_target(
                plan,
                "embed_tokens.weight",
            )
            lm_head = _binding_by_target(plan, "lm_head.weight")
            assert embedding.destination is model.embed_tokens.weight
            assert lm_head.destination is model.lm_head.weight
            assert lm_head.destination is not embedding.destination
            assert lm_head.destination_kind == "parameter"
            assert lm_head.loader_kind == "custom_parameter_loader"
            assert lm_head.local_shape == (
                VOCAB_SIZE // world_size,
                HIDDEN_SIZE,
            )
            assert lm_head.destination_slice is None
            _assert_unchanged(model, pool, snapshot)


def main():
    test_tp_1_and_2_bind_complete_real_component_graph_read_only()
    test_tp4_binds_complete_replicated_kv_heads_by_source_rank()
    test_fail_closed_component_shape_dtype_loader_alias_and_plan_contracts()
    test_meta_embedding_alias_requires_the_same_parameter_object()
    test_untied_lm_head_binds_independent_tp_shard()
    print("qwen35 checkpoint target binding tests passed (4 tests)")


if __name__ == "__main__":
    main()
