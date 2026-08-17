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
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointLoadTarget,
    Qwen35CheckpointSource,
    Qwen35CheckpointTensorLoad,
    Qwen35CheckpointTensorMetadata,
    Qwen35CheckpointTensorPlan,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
)
from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    Qwen35PreparedCheckpointCandidateTarget,
    prepare_qwen35_checkpoint_candidate_target,
)
from tinyvllm.models.qwen35_components import (
    Qwen35ConcreteComponentAssembly,
)


HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 12
VOCAB_SIZE = 32
KEY_HEADS = 2
VALUE_HEADS = 2
KEY_HEAD_DIM = 2
VALUE_HEAD_DIM = 2
FULL_QUERY_HEADS = 2
FULL_KV_HEADS = 2
FULL_HEAD_DIM = 8
CONV_KERNEL = 3
COMPUTE_DTYPE = torch.bfloat16


def _config():
    return types.SimpleNamespace(
        dtype="bfloat16",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=VOCAB_SIZE,
        num_hidden_layers=2,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=KEY_HEADS,
        linear_num_value_heads=VALUE_HEADS,
        linear_key_head_dim=KEY_HEAD_DIM,
        linear_value_head_dim=VALUE_HEAD_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        num_attention_heads=FULL_QUERY_HEADS,
        num_key_value_heads=FULL_KV_HEADS,
        head_dim=FULL_HEAD_DIM,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        rope_parameters={
            "rope_theta": 1_000_000,
            "partial_rotary_factor": 0.75,
            "mrope_section": (1, 1, 1),
        },
    )


def _pool(
    config,
    world_size,
    *,
    recurrent_dtype=None,
    speculative_tokens=1,
):
    layout = build_qwen35_hybrid_state_layout(
        config,
        tensor_parallel_size=world_size,
        dtype=COMPUTE_DTYPE,
        recurrent_dtype=recurrent_dtype,
        speculative_tokens=speculative_tokens,
    )
    return HybridStateTensorPool(layout, capacity=2, device="cpu")


def _tensor_plan() -> Qwen35CheckpointTensorPlan:
    global_key_width = KEY_HEADS * KEY_HEAD_DIM
    global_value_width = VALUE_HEADS * VALUE_HEAD_DIM
    global_conv_width = 2 * global_key_width + global_value_width
    entries = [
        ("embed_tokens.weight", (VOCAB_SIZE, HIDDEN_SIZE), "BF16", None),
        ("final_norm.weight", (HIDDEN_SIZE,), "BF16", None),
    ]
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
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        end = offset + element_count * (2 if dtype == "BF16" else 4)
        loads.append(Qwen35CheckpointTensorLoad(
            weight=Qwen35CheckpointLoadTarget(
                source=Qwen35CheckpointSource(
                    name=f"source.{index:02d}.{target}",
                    shard="model.safetensors",
                ),
                target=target,
                packed_slot=packed_slot,
            ),
            metadata=Qwen35CheckpointTensorMetadata(
                dtype=dtype,
                shape=shape,
                data_offsets=(offset, end),
            ),
            transform=(
                "squeeze_conv_channel"
                if target.endswith("linear_attention.conv_weight")
                else "identity"
            ),
        ))
        offset = end
    return Qwen35CheckpointTensorPlan(
        loads=tuple(loads),
        skips=(),
        payload_bytes=offset,
    )


class _Backend(nn.Module):

    def __init__(self, arguments):
        super().__init__()
        self.arguments = arguments


def _build_backend(layer_index, query_heads, kv_heads, head_dim):
    return _Backend((layer_index, query_heads, kv_heads, head_dim))


def _snapshot_pool(pool):
    return {
        "layout_id": id(pool.layout),
        "capacity": pool.capacity,
        "device": pool.device,
        "bindings": dict(pool._bindings),
        "tensors": {
            key: (
                id(tensor),
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor.clone(),
            )
            for key, tensor in pool._tensors.items()
        },
    }


def _assert_pool_unchanged(pool, snapshot):
    assert id(pool.layout) == snapshot["layout_id"]
    assert pool.capacity == snapshot["capacity"]
    assert pool.device == snapshot["device"]
    assert pool._bindings == snapshot["bindings"]
    assert set(pool._tensors) == set(snapshot["tensors"])
    for key, tensor in pool._tensors.items():
        object_id, pointer, offset, value = snapshot["tensors"][key]
        assert id(tensor) == object_id
        assert tensor.untyped_storage().data_ptr() == pointer
        assert tensor.storage_offset() == offset
        torch.testing.assert_close(tensor, value)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_tp_1_and_2_meta_composition_is_exact_and_one_shot():
    config = _config()
    tensor_plan = _tensor_plan()
    assert len(tensor_plan.loads) == 27
    for world_size in (1, 2):
        for rank in range(world_size):
            pool = _pool(config, world_size)
            snapshot = _snapshot_pool(pool)
            original_init = HybridStateTensorPool.__init__

            def forbidden_pool_init(*_, **__):
                raise AssertionError("factory must not construct a state pool")

            HybridStateTensorPool.__init__ = forbidden_pool_init
            try:
                target = prepare_qwen35_checkpoint_candidate_target(
                    config,
                    tensor_plan,
                    pool=pool,
                    tensor_parallel_size=world_size,
                    tensor_parallel_rank=rank,
                    build_attention_backend=_build_backend,
                    parameter_device="meta",
                )
            finally:
                HybridStateTensorPool.__init__ = original_init

            assert type(target) is Qwen35PreparedCheckpointCandidateTarget
            assert type(target.assembly) is Qwen35ConcreteComponentAssembly
            assert type(target.binding_plan) is Qwen35CheckpointBindingPlan
            assert target.pool is pool
            assert target.assembly.packed.pool is pool
            assert target.assembly.tensor_parallel_size == world_size
            assert target.assembly.tensor_parallel_rank == rank
            assert target.binding_plan.tensor_parallel_size == world_size
            assert target.binding_plan.tensor_parallel_rank == rank
            assert len(target.binding_plan.bindings) == 27
            assert all(
                binding.destination.device.type == "meta"
                for binding in target.binding_plan.bindings
            )

            model = target.assembly.packed.model
            candidate = target.take()
            assert candidate == (model, target.binding_plan)
            assert candidate[0] is model
            assert candidate[1] is target.binding_plan
            _expect_error(target.take, "already consumed")
            _assert_pool_unchanged(pool, snapshot)


def test_compact_cpu_composition_preserves_pool_and_destination_identity():
    config = _config()
    pool = _pool(config, 1)
    snapshot = _snapshot_pool(pool)
    target = prepare_qwen35_checkpoint_candidate_target(
        config,
        _tensor_plan(),
        pool=pool,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        build_attention_backend=_build_backend,
        parameter_device="cpu",
    )

    model, binding_plan = target.take()
    registered = dict(
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    assert all(
        binding.destination.device.type == "cpu"
        for binding in binding_plan.bindings
    )
    assert all(
        binding.destination is registered[binding.destination_name]
        for binding in binding_plan.bindings
    )
    _assert_pool_unchanged(pool, snapshot)


def test_cpu_composition_preserves_mixed_state_dtypes_in_model_adapters():
    config = _config()
    pool = _pool(
        config,
        1,
        recurrent_dtype=torch.float32,
    )
    snapshot = _snapshot_pool(pool)
    target = prepare_qwen35_checkpoint_candidate_target(
        config,
        _tensor_plan(),
        pool=pool,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        build_attention_backend=_build_backend,
        parameter_device="cpu",
    )

    convolution = pool.component_tensor(0, "linear_convolution")
    recurrent = pool.component_tensor(0, "linear_recurrent")
    assert convolution.dtype == torch.bfloat16
    assert recurrent.dtype == torch.float32
    assert target.assembly.packed.pool is pool
    assert target.assembly.packed.state_transaction.pool is pool
    assert (
        target.assembly.packed.model.layer_stack.state_transaction
        is target.assembly.packed.state_transaction
    )
    assert len(target.assembly.packed.adapters) == 1
    adapter = target.assembly.packed.adapters[0]
    assert adapter.pool is pool
    assert adapter.convolution is convolution
    assert adapter.recurrent is recurrent
    assert adapter.convolution.dtype == torch.bfloat16
    assert adapter.recurrent.dtype == torch.float32
    _assert_pool_unchanged(pool, snapshot)


def test_backend_failure_returns_no_target_and_preserves_pool():
    config = _config()
    pool = _pool(config, 1)
    snapshot = _snapshot_pool(pool)

    def fail_backend(*_):
        raise RuntimeError("backend construction failed")

    _expect_error(
        lambda: prepare_qwen35_checkpoint_candidate_target(
            config,
            _tensor_plan(),
            pool=pool,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            build_attention_backend=fail_backend,
        ),
        "backend construction failed",
    )
    _assert_pool_unchanged(pool, snapshot)


def test_binding_failure_returns_no_target_and_preserves_pool():
    config = _config()
    pool = _pool(config, 1)
    snapshot = _snapshot_pool(pool)
    tensor_plan = _tensor_plan()
    bad_load = replace(
        tensor_plan.loads[0],
        weight=replace(
            tensor_plan.loads[0].weight,
            target="embed_tokens.missing",
        ),
    )
    bad_plan = replace(
        tensor_plan,
        loads=(bad_load, *tensor_plan.loads[1:]),
    )

    _expect_error(
        lambda: prepare_qwen35_checkpoint_candidate_target(
            config,
            bad_plan,
            pool=pool,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            build_attention_backend=_build_backend,
        ),
        "missing",
    )
    _assert_pool_unchanged(pool, snapshot)


def test_malformed_plan_and_pool_tp_mismatch_preserve_pool():
    config = _config()
    tensor_plan = _tensor_plan()

    pool = _pool(config, 1)
    snapshot = _snapshot_pool(pool)
    malformed_plan = replace(
        tensor_plan,
        loads=(object(), *tensor_plan.loads[1:]),
    )
    _expect_error(
        lambda: prepare_qwen35_checkpoint_candidate_target(
            config,
            malformed_plan,
            pool=pool,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            build_attention_backend=_build_backend,
        ),
        "Qwen35CheckpointTensorLoad",
    )
    _assert_pool_unchanged(pool, snapshot)

    pool = _pool(config, 1)
    snapshot = _snapshot_pool(pool)
    _expect_error(
        lambda: prepare_qwen35_checkpoint_candidate_target(
            config,
            tensor_plan,
            pool=pool,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
            build_attention_backend=_build_backend,
        ),
        "pool layout",
    )
    _assert_pool_unchanged(pool, snapshot)


def test_factory_accepts_valid_wider_speculative_state_layout():
    config = _config()
    pool = _pool(config, 2, speculative_tokens=3)
    snapshot = _snapshot_pool(pool)

    target = prepare_qwen35_checkpoint_candidate_target(
        config,
        _tensor_plan(),
        pool=pool,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        build_attention_backend=_build_backend,
    )

    assert target.pool is pool
    assert target.assembly.packed.pool is pool
    assert len(target.binding_plan.bindings) == 27
    _assert_pool_unchanged(pool, snapshot)


def main():
    test_tp_1_and_2_meta_composition_is_exact_and_one_shot()
    test_compact_cpu_composition_preserves_pool_and_destination_identity()
    test_cpu_composition_preserves_mixed_state_dtypes_in_model_adapters()
    test_backend_failure_returns_no_target_and_preserves_pool()
    test_binding_failure_returns_no_target_and_preserves_pool()
    test_malformed_plan_and_pool_tp_mismatch_preserve_pool()
    test_factory_accepts_valid_wider_speculative_state_layout()
    print("qwen35 checkpoint candidate factory tests passed (7 tests)")


if __name__ == "__main__":
    main()
