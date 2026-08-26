from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "qwen35_checkpoint_target_binding_helper",
        ROOT / "tools/test_qwen35_checkpoint_target_binding.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()

from tinyvllm.models.qwen35_checkpoint_assignment import (
    Qwen35CheckpointAssignmentResult,
    _assign_qwen35_checkpoint_source_bindings,
    _direct_buffer_local,
    _prepare_operation,
    assign_qwen35_checkpoint_tensors,
)
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointLoadTarget,
    Qwen35CheckpointSource,
    Qwen35CheckpointTensorLoad,
    Qwen35CheckpointTensorMetadata,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointTensorBinding,
)


def _source_tensors(tensor_plan):
    sources = {}
    for source_index, load in enumerate(tensor_plan.loads):
        dtype = (
            torch.bfloat16
            if load.metadata.dtype == "BF16"
            else torch.float32
        )
        element_count = 1
        for dimension in load.metadata.shape:
            element_count *= dimension
        values = (
            torch.arange(element_count, dtype=torch.float32)
            .reshape(load.metadata.shape)
            + source_index * 100
        )
        sources[load.weight.source.name] = values.to(dtype)
    return sources


def _unique_destinations(binding_plan):
    destinations = {}
    for binding in binding_plan.bindings:
        destinations.setdefault(id(binding.destination), binding.destination)
    return destinations


def _initialize_destinations(binding_plan):
    with torch.no_grad():
        for destination in _unique_destinations(binding_plan).values():
            destination.fill_(-7)


def _snapshot_destinations(binding_plan):
    return {
        object_id: (
            destination,
            destination.detach().clone(),
        )
        for object_id, destination in _unique_destinations(
            binding_plan
        ).items()
    }


def _assert_destinations_equal(snapshot):
    for destination, expected in snapshot.values():
        torch.testing.assert_close(destination, expected)


def _snapshot_sources(sources):
    return {
        name: (id(tensor), tensor.clone())
        for name, tensor in sources.items()
    }


def _assert_sources_unchanged(sources, snapshot):
    assert set(sources) == set(snapshot)
    for name, tensor in sources.items():
        object_id, expected = snapshot[name]
        assert id(tensor) == object_id
        torch.testing.assert_close(tensor, expected)


def _snapshot_registrations(model):
    return tuple(
        (name, id(tensor), tensor.shape, tensor.dtype, tensor.device)
        for name, tensor in (
            list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False))
        )
    )


def _transformed(binding, source):
    if binding.load.transform == "identity":
        return source
    if binding.load.transform == "squeeze_conv_channel":
        return source.squeeze(1)
    raise AssertionError(binding.load.transform)


def _expected_local(binding, source, world_size, rank):
    transformed = _transformed(binding, source)
    target = binding.load.weight.target
    if binding.loader_kind == "default_parameter_copy":
        return transformed
    if binding.loader_kind == "direct_buffer_copy":
        if target.endswith("linear_attention.norm_weight"):
            return transformed.to(binding.destination.dtype)
        if binding.source_segments is not None:
            shards = []
            offset = 0
            for global_rows in binding.source_segments:
                local_rows = global_rows // world_size
                shards.append(
                    transformed.narrow(
                        0,
                        offset + rank * local_rows,
                        local_rows,
                    )
                )
                offset += global_rows
            return torch.cat(shards, dim=0)
        local_rows = transformed.shape[0] // world_size
        return transformed.narrow(0, rank * local_rows, local_rows)
    if target in ("embed_tokens.weight", "lm_head.weight"):
        local_rows = transformed.shape[0] // world_size
        return transformed.narrow(0, rank * local_rows, local_rows)
    if target.endswith("mlp.gate_up_proj.weight"):
        return transformed
    if target.endswith("mlp.down_proj.weight"):
        return transformed
    if target.endswith((
        "full_attention.q_projection.weight",
        "full_attention.k_projection.weight",
        "full_attention.v_projection.weight",
    )):
        return transformed
    if target.endswith((
        "linear_attention.out_proj.weight",
        "full_attention.output_projection.weight",
    )):
        local_columns = transformed.shape[1] // world_size
        return transformed.narrow(
            1,
            rank * local_columns,
            local_columns,
        )
    if target.endswith("linear_attention.in_proj_qkv.weight"):
        output_sizes = (
            helper.KEY_HEADS * helper.KEY_HEAD_DIM,
            helper.KEY_HEADS * helper.KEY_HEAD_DIM,
            helper.VALUE_HEADS * helper.VALUE_HEAD_DIM,
        )
        shards = []
        offset = 0
        for output_size in output_sizes:
            local_rows = output_size // world_size
            segment = transformed.narrow(0, offset, output_size)
            shards.append(
                segment.narrow(0, rank * local_rows, local_rows)
            )
            offset += output_size
        return torch.cat(shards)
    if target.endswith((
        "linear_attention.in_proj_b.weight",
        "linear_attention.in_proj_a.weight",
    )):
        return transformed
    local_rows = transformed.shape[0] // world_size
    return transformed.narrow(0, rank * local_rows, local_rows)


def _expected_destinations(binding_plan, sources):
    expected = {
        object_id: torch.full_like(destination, -7)
        for object_id, destination in _unique_destinations(
            binding_plan
        ).items()
    }
    world_size = binding_plan.tensor_parallel_size
    rank = binding_plan.tensor_parallel_rank
    for binding in binding_plan.bindings:
        source = sources[binding.load.weight.source.name]
        local = _expected_local(
            binding,
            source,
            world_size,
            rank,
        )
        destination = expected[id(binding.destination)]
        if binding.destination_slice is None:
            destination.copy_(local)
        else:
            offset, length = binding.destination_slice
            destination.narrow(0, offset, length).copy_(local)
    return expected


def _fixture(rank, world_size, *, tie_word_embeddings=True):
    model, pool = helper._fixture(
        rank,
        world_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    tensor_plan = helper._tensor_plan(
        tie_word_embeddings=tie_word_embeddings,
    )
    binding_plan = helper.build_qwen35_checkpoint_binding_plan(
        model,
        tensor_plan,
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
    )
    return model, pool, tensor_plan, binding_plan


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_tp_1_and_2_assign_all_bindings_exactly():
    for world_size in (1, 2):
        for rank in range(world_size):
            model, pool, tensor_plan, binding_plan = _fixture(
                rank,
                world_size,
            )
            _initialize_destinations(binding_plan)
            sources = _source_tensors(tensor_plan)
            expected = _expected_destinations(binding_plan, sources)
            source_snapshot = _snapshot_sources(sources)
            pool_snapshot = {
                key: (id(tensor), tensor.clone())
                for key, tensor in pool._tensors.items()
            }
            registrations = _snapshot_registrations(model)

            result = assign_qwen35_checkpoint_tensors(
                binding_plan,
                sources,
            )

            assert type(result) is Qwen35CheckpointAssignmentResult
            assert result.assigned_bindings == 27
            assert result.unique_destinations == 25
            assert result.source_tensors == 27
            for object_id, destination in _unique_destinations(
                binding_plan
            ).items():
                torch.testing.assert_close(
                    destination,
                    expected[object_id],
                )
            assert (
                model.embed_tokens.weight.untyped_storage().data_ptr()
                == model.lm_head.weight.untyped_storage().data_ptr()
            )
            assert (
                model.embed_tokens.weight.storage_offset()
                == model.lm_head.weight.storage_offset()
            )
            assert _snapshot_registrations(model) == registrations
            _assert_sources_unchanged(sources, source_snapshot)
            for key, tensor in pool._tensors.items():
                object_id, value = pool_snapshot[key]
                assert id(tensor) == object_id
                torch.testing.assert_close(tensor, value)


def test_bf16_a_log_is_explicitly_cast_to_float32_destination():
    model, _, tensor_plan, _ = _fixture(0, 1)
    loads = list(tensor_plan.loads)
    a_log_index = next(
        index
        for index, load in enumerate(loads)
        if load.weight.target.endswith("linear_attention.A_log")
    )
    loads[a_log_index] = replace(
        loads[a_log_index],
        metadata=replace(
            loads[a_log_index].metadata,
            dtype="BF16",
        ),
    )
    qwen38_tensor_plan = replace(
        tensor_plan,
        loads=tuple(loads),
    )
    binding_plan = helper.build_qwen35_checkpoint_binding_plan(
        model,
        qwen38_tensor_plan,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    binding = helper._binding_by_target(
        binding_plan,
        "layers.0.linear_attention.A_log",
    )
    source = torch.tensor((1.25, -2.5), dtype=torch.bfloat16)

    operation = _prepare_operation(binding, source, 1, 0)

    assert operation.source.dtype == torch.bfloat16
    assert operation.transformed.dtype == torch.float32
    assert operation.local_tensor.dtype == torch.float32
    _assign_qwen35_checkpoint_source_bindings(
        (binding,),
        source,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    torch.testing.assert_close(
        binding.destination,
        source.to(torch.float32),
    )


def test_tp_1_and_2_assign_independent_lm_head_exactly():
    for world_size in (1, 2):
        for rank in range(world_size):
            model, _, tensor_plan, binding_plan = _fixture(
                rank,
                world_size,
                tie_word_embeddings=False,
            )
            _initialize_destinations(binding_plan)
            sources = _source_tensors(tensor_plan)

            result = assign_qwen35_checkpoint_tensors(
                binding_plan,
                sources,
            )

            lm_head = helper._binding_by_target(
                binding_plan,
                "lm_head.weight",
            )
            embedding = helper._binding_by_target(
                binding_plan,
                "embed_tokens.weight",
            )
            assert result.assigned_bindings == 28
            assert lm_head.destination is not embedding.destination
            lm_head_source = sources[lm_head.load.weight.source.name]
            embedding_source = sources[
                embedding.load.weight.source.name
            ]
            local_rows = helper.VOCAB_SIZE // world_size
            torch.testing.assert_close(
                lm_head.destination,
                lm_head_source.narrow(
                    0,
                    rank * local_rows,
                    local_rows,
                ),
            )
            torch.testing.assert_close(
                embedding.destination,
                embedding_source.narrow(
                    0,
                    rank * local_rows,
                    local_rows,
                ),
            )


def test_tp2_output_projections_assign_matching_local_columns():
    world_size = 2
    targets = (
        "layers.0.linear_attention.out_proj.weight",
        "layers.1.full_attention.output_projection.weight",
    )
    for rank in range(world_size):
        _, _, tensor_plan, binding_plan = _fixture(
            rank,
            world_size,
        )
        sources = _source_tensors(tensor_plan)
        assign_qwen35_checkpoint_tensors(binding_plan, sources)
        for target in targets:
            binding = helper._binding_by_target(
                binding_plan,
                target,
            )
            source = sources[binding.load.weight.source.name]
            transformed = _transformed(binding, source)
            local_columns = transformed.shape[1] // world_size
            expected = transformed.narrow(
                1,
                rank * local_columns,
                local_columns,
            )
            assert binding.destination.shape == expected.shape
            assert torch.equal(binding.destination, expected)


def test_tp4_assignment_replicates_complete_kv_projection_weights():
    tensor_plan = helper._tp4_replication_tensor_plan()
    rank_payloads = []
    sources = _source_tensors(tensor_plan)
    source_snapshot = _snapshot_sources(sources)
    for rank in range(4):
        model, pool = helper._tp4_replication_fixture(rank)
        binding_plan = helper.build_qwen35_checkpoint_binding_plan(
            model,
            tensor_plan,
            tensor_parallel_size=4,
            tensor_parallel_rank=rank,
        )
        _initialize_destinations(binding_plan)
        result = assign_qwen35_checkpoint_tensors(
            binding_plan,
            sources,
        )
        assert result.assigned_bindings == 2
        payload = {}
        for suffix in ("k_projection.weight", "v_projection.weight"):
            binding = helper._binding_by_target(
                binding_plan,
                f"layers.1.full_attention.{suffix}",
            )
            payload[suffix] = binding.destination.detach().clone()
            source = sources[binding.load.weight.source.name]
            assert binding.destination.shape == source.shape
            torch.testing.assert_close(payload[suffix], source)
        rank_payloads.append(payload)
        assert all(
            tensor.device.type == "cpu"
            for tensor in pool._tensors.values()
        )

    for suffix in ("k_projection.weight", "v_projection.weight"):
        torch.testing.assert_close(
            rank_payloads[0][suffix],
            rank_payloads[1][suffix],
        )
        torch.testing.assert_close(
            rank_payloads[0][suffix],
            rank_payloads[2][suffix],
        )
        torch.testing.assert_close(
            rank_payloads[0][suffix],
            rank_payloads[3][suffix],
        )
    _assert_sources_unchanged(sources, source_snapshot)


def test_tp2_assignment_replicates_complete_linear_attention_gate_weights():
    destinations_by_rank = []
    for rank in range(2):
        _, _, tensor_plan, binding_plan = _fixture(rank, 2)
        _initialize_destinations(binding_plan)
        sources = _source_tensors(tensor_plan)

        assign_qwen35_checkpoint_tensors(binding_plan, sources)

        rank_destinations = {}
        for suffix in ("in_proj_b.weight", "in_proj_a.weight"):
            target = f"layers.0.linear_attention.{suffix}"
            binding = helper._binding_by_target(binding_plan, target)
            source = sources[binding.load.weight.source.name]
            assert binding.destination.shape == source.shape
            torch.testing.assert_close(binding.destination, source)
            rank_destinations[suffix] = (
                binding.destination.detach().clone()
            )
        destinations_by_rank.append(rank_destinations)

    for suffix in ("in_proj_b.weight", "in_proj_a.weight"):
        torch.testing.assert_close(
            destinations_by_rank[0][suffix],
            destinations_by_rank[1][suffix],
        )


def test_conv_weight_channels_match_segmented_qkv_projection_order():
    world_size = 2
    output_sizes = (
        helper.KEY_HEADS * helper.KEY_HEAD_DIM,
        helper.KEY_HEADS * helper.KEY_HEAD_DIM,
        helper.VALUE_HEADS * helper.VALUE_HEAD_DIM,
    )
    for rank in range(world_size):
        _, _, tensor_plan, binding_plan = _fixture(rank, world_size)
        sources = _source_tensors(tensor_plan)
        assign_qwen35_checkpoint_tensors(binding_plan, sources)

        projection = helper._binding_by_target(
            binding_plan,
            "layers.0.linear_attention.in_proj_qkv.weight",
        )
        convolution = helper._binding_by_target(
            binding_plan,
            "layers.0.linear_attention.conv_weight",
        )
        projection_source = sources[projection.load.weight.source.name]
        convolution_source = sources[convolution.load.weight.source.name]

        expected_projection_rows = []
        expected_convolution_rows = []
        global_offset = 0
        for output_size in output_sizes:
            local_rows = output_size // world_size
            source_start = global_offset + rank * local_rows
            expected_projection_rows.append(
                projection_source.narrow(
                    0,
                    source_start,
                    local_rows,
                )
            )
            expected_convolution_rows.append(
                convolution_source.squeeze(1).narrow(
                    0,
                    source_start,
                    local_rows,
                )
            )
            global_offset += output_size

        assert torch.equal(
            projection.destination,
            torch.cat(expected_projection_rows, dim=0),
        )
        assert torch.equal(
            convolution.destination,
            torch.cat(expected_convolution_rows, dim=0),
        )


def test_tp4_direct_conv_buffer_uses_segmented_channel_shards():
    source_segments = (8, 8, 16)
    source = torch.arange(
        sum(source_segments) * 3,
        dtype=torch.bfloat16,
    ).reshape(sum(source_segments), 3)
    binding = Qwen35CheckpointTensorBinding(
        load=Qwen35CheckpointTensorLoad(
            weight=Qwen35CheckpointLoadTarget(
                source=Qwen35CheckpointSource(
                    name="linear_attn.conv1d.weight",
                    shard="model.safetensors",
                ),
                target="layers.0.linear_attention.conv_weight",
                packed_slot=None,
            ),
            metadata=Qwen35CheckpointTensorMetadata(
                dtype="BF16",
                shape=(sum(source_segments), 1, 3),
                data_offsets=(0, source.numel() * source.element_size()),
            ),
            transform="squeeze_conv_channel",
        ),
        destination_name=(
            "layer_stack.layers.0.linear_attention.conv_weight"
        ),
        destination=torch.empty(8, 3, dtype=torch.bfloat16),
        destination_kind="buffer",
        loader_kind="direct_buffer_copy",
        local_shape=(8, 3),
        destination_slice=None,
        source_segments=source_segments,
    )
    for rank in range(4):
        expected = []
        offset = 0
        for global_rows in source_segments:
            local_rows = global_rows // 4
            expected.append(
                source.narrow(
                    0,
                    offset + rank * local_rows,
                    local_rows,
                )
            )
            offset += global_rows
        actual = _direct_buffer_local(
            binding,
            source,
            tensor_parallel_size=4,
            tensor_parallel_rank=rank,
        )
        assert torch.equal(actual, torch.cat(expected, dim=0))


def test_prevalidation_failures_do_not_mutate_destinations():
    model, _, tensor_plan, binding_plan = _fixture(0, 1)
    _initialize_destinations(binding_plan)
    sources = _source_tensors(tensor_plan)

    cases = []
    cases.append((
        object(),
        sources,
        "exact Qwen35CheckpointBindingPlan",
    ))
    cases.append((binding_plan, [], "source_tensors must be a mapping"))
    missing = dict(sources)
    missing.pop(next(iter(missing)))
    cases.append((binding_plan, missing, "source coverage"))
    extra = dict(sources)
    extra["unexpected"] = torch.ones(1)
    cases.append((binding_plan, extra, "source coverage"))
    non_tensor = dict(sources)
    non_tensor[next(iter(non_tensor))] = object()
    cases.append((binding_plan, non_tensor, "must be a tensor"))

    first_name = binding_plan.bindings[0].load.weight.source.name
    wrong_shape = dict(sources)
    wrong_shape[first_name] = wrong_shape[first_name][:-1]
    cases.append((binding_plan, wrong_shape, "shape"))
    wrong_dtype = dict(sources)
    wrong_dtype[first_name] = wrong_dtype[first_name].float()
    cases.append((binding_plan, wrong_dtype, "dtype"))
    wrong_device = dict(sources)
    wrong_device[first_name] = torch.empty(
        sources[first_name].shape,
        dtype=sources[first_name].dtype,
        device="meta",
    )
    cases.append((binding_plan, wrong_device, "CPU"))

    bad_transform_binding = replace(
        binding_plan.bindings[0],
        load=replace(
            binding_plan.bindings[0].load,
            transform="unknown",
        ),
    )
    cases.append((
        replace(
            binding_plan,
            bindings=(
                bad_transform_binding,
                *binding_plan.bindings[1:],
            ),
        ),
        sources,
        "unsupported checkpoint transform",
    ))
    bad_loader_binding = replace(
        binding_plan.bindings[0],
        loader_kind="unknown",
    )
    cases.append((
        replace(
            binding_plan,
            bindings=(
                bad_loader_binding,
                *binding_plan.bindings[1:],
            ),
        ),
        sources,
        "unsupported loader kind",
    ))
    meta_binding = replace(
        binding_plan.bindings[0],
        destination=torch.empty(
            binding_plan.bindings[0].destination.shape,
            dtype=binding_plan.bindings[0].destination.dtype,
            device="meta",
        ),
    )
    cases.append((
        replace(
            binding_plan,
            bindings=(
                meta_binding,
                *binding_plan.bindings[1:],
            ),
        ),
        sources,
        "destination must be a CPU tensor",
    ))

    for case_plan, case_sources, message in cases:
        snapshot = _snapshot_destinations(binding_plan)
        _expect_error(
            lambda case_plan=case_plan, case_sources=case_sources: (
                assign_qwen35_checkpoint_tensors(
                    case_plan,
                    case_sources,
                )
            ),
            message,
        )
        _assert_destinations_equal(snapshot)

    embedding = model.embed_tokens.weight
    original_loader = embedding.weight_loader
    delattr(embedding, "weight_loader")
    try:
        snapshot = _snapshot_destinations(binding_plan)
        _expect_error(
            lambda: assign_qwen35_checkpoint_tensors(
                binding_plan,
                sources,
            ),
            "callable custom loader",
        )
        _assert_destinations_equal(snapshot)
    finally:
        embedding.weight_loader = original_loader


def test_mid_assignment_failure_rolls_back_every_destination():
    model, _, tensor_plan, binding_plan = _fixture(0, 2)
    _initialize_destinations(binding_plan)
    sources = _source_tensors(tensor_plan)
    destination_snapshot = _snapshot_destinations(binding_plan)
    source_snapshot = _snapshot_sources(sources)
    registrations = _snapshot_registrations(model)

    failing_binding = next(
        binding
        for binding in reversed(binding_plan.bindings)
        if binding.loader_kind == "custom_parameter_loader"
    )
    destination = failing_binding.destination
    original_loader = destination.weight_loader

    def failing_loader(*_):
        raise RuntimeError("injected assignment failure")

    destination.weight_loader = failing_loader
    try:
        _expect_error(
            lambda: assign_qwen35_checkpoint_tensors(
                binding_plan,
                sources,
            ),
            failing_binding.load.weight.source.name,
        )
    finally:
        destination.weight_loader = original_loader

    _assert_destinations_equal(destination_snapshot)
    _assert_sources_unchanged(sources, source_snapshot)
    assert _snapshot_registrations(model) == registrations
    assert (
        model.embed_tokens.weight.untyped_storage().data_ptr()
        == model.lm_head.weight.untyped_storage().data_ptr()
    )
    assert (
        model.embed_tokens.weight.storage_offset()
        == model.lm_head.weight.storage_offset()
    )


def test_single_source_assignment_group_contract():
    _, _, tensor_plan, binding_plan = _fixture(0, 2)
    _initialize_destinations(binding_plan)
    sources = _source_tensors(tensor_plan)
    packed_bindings = tuple(
        binding
        for binding in binding_plan.bindings
        if binding.load.weight.target.endswith(
            "mlp.gate_up_proj.weight"
        )
    )
    assert len(packed_bindings) >= 2
    gate_binding, up_binding = packed_bindings[:2]
    assert gate_binding.destination is up_binding.destination

    for binding in (gate_binding, up_binding):
        assigned = _assign_qwen35_checkpoint_source_bindings(
            (binding,),
            sources[binding.load.weight.source.name],
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
        )
        assert assigned == 1

    expected = _expected_destinations(binding_plan, sources)
    torch.testing.assert_close(
        gate_binding.destination,
        expected[id(gate_binding.destination)],
    )

    cases = (
        (
            (),
            sources[gate_binding.load.weight.source.name],
            2,
            0,
            "non-empty tuple",
        ),
        (
            (gate_binding, up_binding),
            sources[gate_binding.load.weight.source.name],
            2,
            0,
            "one checkpoint source",
        ),
        (
            (gate_binding,),
            sources[gate_binding.load.weight.source.name][:-1],
            2,
            0,
            "shape",
        ),
        (
            (gate_binding,),
            sources[gate_binding.load.weight.source.name].float(),
            2,
            0,
            "dtype",
        ),
        (
            (gate_binding,),
            torch.empty(
                sources[gate_binding.load.weight.source.name].shape,
                dtype=sources[gate_binding.load.weight.source.name].dtype,
                device="meta",
            ),
            2,
            0,
            "CPU",
        ),
        (
            (gate_binding,),
            sources[gate_binding.load.weight.source.name],
            0,
            0,
            "tensor_parallel_size",
        ),
        (
            (gate_binding,),
            sources[gate_binding.load.weight.source.name],
            2,
            2,
            "tensor_parallel_rank",
        ),
    )
    for bindings, source, size, rank, message in cases:
        _expect_error(
            lambda bindings=bindings, source=source, size=size, rank=rank: (
                _assign_qwen35_checkpoint_source_bindings(
                    bindings,
                    source,
                    tensor_parallel_size=size,
                    tensor_parallel_rank=rank,
                )
            ),
            message,
        )

    original_loader = gate_binding.destination.weight_loader

    def failing_loader(*_):
        raise RuntimeError("injected source-group failure")

    gate_binding.destination.weight_loader = failing_loader
    try:
        _expect_error(
            lambda: _assign_qwen35_checkpoint_source_bindings(
                (gate_binding,),
                sources[gate_binding.load.weight.source.name],
                tensor_parallel_size=2,
                tensor_parallel_rank=0,
            ),
            gate_binding.load.weight.source.name,
        )
    finally:
        gate_binding.destination.weight_loader = original_loader


def main():
    test_tp_1_and_2_assign_all_bindings_exactly()
    test_tp4_assignment_replicates_complete_kv_projection_weights()
    test_conv_weight_channels_match_segmented_qkv_projection_order()
    test_tp4_direct_conv_buffer_uses_segmented_channel_shards()
    test_prevalidation_failures_do_not_mutate_destinations()
    test_mid_assignment_failure_rolls_back_every_destination()
    test_single_source_assignment_group_contract()
    print("qwen35 checkpoint assignment tests passed (7 tests)")


if __name__ == "__main__":
    main()
