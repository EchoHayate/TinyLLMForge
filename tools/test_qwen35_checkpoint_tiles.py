from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assignment_helper = _load_helper(
    "qwen35_checkpoint_assignment_test_helper",
    "tools/test_qwen35_checkpoint_assignment.py",
)

from tinyvllm.models.qwen35_checkpoint_tiles import (
    Qwen35CheckpointTile,
    Qwen35CheckpointTilePlan,
    build_qwen35_checkpoint_tile_plan,
)


def _expect_error(function, message):
    try:
        function()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _indices(item, size):
    if isinstance(item, int):
        index = item if item >= 0 else size + item
        return (index,)
    return tuple(range(*item.indices(size)))


def _coordinates(shape, slices):
    axes = [
        _indices(item, size)
        for item, size in zip(slices, shape)
    ]
    if len(shape) == 1:
        return {(row,) for row in axes[0]}
    if len(shape) == 2:
        return {
            (row, column)
            for row in axes[0]
            for column in axes[1]
        }
    raise AssertionError(shape)


def _expected_destination_coordinates(binding):
    shape = tuple(binding.destination.shape)
    if binding.destination_slice is None:
        slices = tuple(slice(0, size) for size in shape)
    else:
        offset, length = binding.destination_slice
        slices = (
            slice(offset, offset + length),
            *tuple(slice(0, size) for size in shape[1:]),
        )
    return _coordinates(shape, slices)


def _assert_exact_destination_coverage(binding_plan, tile_plan):
    by_binding = {
        index: set()
        for index in range(len(binding_plan.bindings))
    }
    for tile in tile_plan.tiles:
        coordinates = _coordinates(
            tuple(tile.destination.shape),
            tile.destination_slices,
        )
        assert not (by_binding[tile.binding_index] & coordinates)
        by_binding[tile.binding_index].update(coordinates)
    for index, binding in enumerate(binding_plan.bindings):
        assert by_binding[index] == _expected_destination_coordinates(
            binding
        )


def _kind_counts(tile_plan):
    return {
        kind: len({
            tile.binding_index
            for tile in tile_plan.tiles
            if tile.kind == kind
        })
        for kind in {
            tile.kind
            for tile in tile_plan.tiles
        }
    }


def _tile_shape_from_slices(tile):
    dimensions = []
    for item, size in zip(
        tile.source_slices,
        tile.source_tensor_shape,
    ):
        if isinstance(item, int):
            continue
        dimensions.append(len(_indices(item, size)))
    return tuple(dimensions)


def test_plans_all_five_binding_classes_at_tp_1_and_2():
    expected_kinds = {
        "axis0": 3,
        "axis1": 2,
        "squeeze_axis0": 1,
        "replicated": 21,
    }
    budgets = {1: 24, 2: 24}
    for world_size in (1, 2):
        for rank in range(world_size):
            _, _, _, binding_plan = assignment_helper._fixture(
                rank,
                world_size,
            )
            result = build_qwen35_checkpoint_tile_plan(
                binding_plan,
                max_tile_bytes=budgets[world_size],
            )

            assert type(result) is Qwen35CheckpointTilePlan
            assert result.tensor_parallel_size == world_size
            assert result.tensor_parallel_rank == rank
            assert result.binding_count == 27
            assert result.source_count == 27
            assert result.peak_tile_bytes <= budgets[world_size]
            assert result.destination_bytes == sum(
                tile.byte_count
                for tile in result.tiles
            )
            assert _kind_counts(result) == expected_kinds
            assert tuple(
                (tile.binding_index, tile.target)
                for tile in result.tiles
            ) == tuple(sorted(
                (
                    tile.binding_index,
                    tile.target,
                )
                for tile in result.tiles
            ))

            for tile in result.tiles:
                assert type(tile) is Qwen35CheckpointTile
                assert tile.destination is (
                    binding_plan.bindings[
                        tile.binding_index
                    ].destination
                )
                assert tile.tile_shape == _tile_shape_from_slices(tile)
                element_count = 1
                for dimension in tile.tile_shape:
                    element_count *= dimension
                assert tile.byte_count == (
                    element_count
                    * torch.empty(
                        (),
                        dtype=tile.dtype,
                    ).element_size()
                )
                assert 0 < tile.byte_count <= budgets[world_size]
                assert _coordinates(
                    tuple(tile.destination.shape),
                    tile.destination_slices,
                )
            _assert_exact_destination_coverage(
                binding_plan,
                result,
            )


def test_representative_source_and_destination_slices():
    _, _, _, binding_plan = assignment_helper._fixture(1, 2)
    result = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=24,
    )

    embedding_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target == "embed_tokens.weight"
    )
    embedding_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == embedding_index
    ]
    assert embedding_tiles[0].source_slices == (
        slice(16, 17),
        slice(0, 8),
    )
    assert embedding_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 8),
    )

    row_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "mlp.down_proj.weight"
        )
    )
    row_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == row_index
    ]
    assert row_tiles[0].kind == "replicated"
    assert row_tiles[0].source_slices == (
        slice(0, 1),
        slice(0, 12),
    )
    assert row_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 12),
    )

    replicated_gate_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "linear_attention.in_proj_a.weight"
        )
    )
    replicated_gate_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == replicated_gate_index
    ]
    assert replicated_gate_tiles[0].kind == "replicated"
    assert replicated_gate_tiles[0].source_slices == (
        slice(0, 1),
        slice(0, 8),
    )
    assert replicated_gate_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 8),
    )


    query_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "full_attention.q_projection.weight"
        )
    )
    query_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == query_index
    ]
    assert query_tiles[0].kind == "replicated"
    assert query_tiles[0].source_slices == (
        slice(0, 1),
        slice(0, 8),
    )
    assert query_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 8),
    )

    qkv_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "linear_attention.in_proj_qkv.weight"
        )
    )
    qkv_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == qkv_index
    ]
    assert qkv_tiles[0].kind == "replicated"
    assert qkv_tiles[0].source_slices == (
        slice(0, 1),
        slice(0, 8),
    )
    assert qkv_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 8),
    )
    assert any(
        tile.source_slices[0] == slice(10, 11)
        and tile.destination_slices[0] == slice(10, 11)
        for tile in qkv_tiles
    )

    z_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "linear_attention.in_proj_z.weight"
        )
    )
    z_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == z_index
    ]
    assert z_tiles[0].kind == "replicated"
    assert z_tiles[0].source_slices == (
        slice(0, 1),
        slice(0, 8),
    )
    assert z_tiles[0].destination_slices == (
        slice(0, 1),
        slice(0, 8),
    )

    output_projection_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "full_attention.output_projection.weight"
        )
    )
    output_projection_tiles = [
        tile
        for tile in result.tiles
        if tile.binding_index == output_projection_index
    ]
    assert output_projection_tiles[0].kind == "axis1"
    assert output_projection_tiles[0].source_slices == (
        slice(0, 6),
        slice(2, 4),
    )
    assert output_projection_tiles[0].destination_slices == (
        slice(0, 6),
        slice(0, 2),
    )

    convolution_index = next(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.endswith(
            "linear_attention.conv_weight"
        )
    )
    convolution_tile = next(
        tile
        for tile in result.tiles
        if tile.binding_index == convolution_index
    )
    assert convolution_tile.source_slices == (
        slice(2, 4),
        0,
        slice(0, 3),
    )
    assert convolution_tile.tile_shape == (2, 3)
    assert convolution_tile.destination_slices == (
        slice(0, 2),
        slice(0, 3),
    )


def test_untied_lm_head_uses_axis_zero_tp_tiles():
    for rank in range(2):
        _, _, _, binding_plan = assignment_helper._fixture(
            rank,
            2,
            tie_word_embeddings=False,
        )
        lm_head = next(
            binding
            for binding in binding_plan.bindings
            if binding.load.weight.target == "lm_head.weight"
        )
        isolated_plan = replace(
            binding_plan,
            bindings=(lm_head,),
        )
        lm_head_tiles = build_qwen35_checkpoint_tile_plan(
            isolated_plan,
            max_tile_bytes=24,
        ).tiles

        assert lm_head_tiles
        assert all(tile.kind == "axis0" for tile in lm_head_tiles)
        assert lm_head_tiles[0].source_slices == (
            slice(rank * 16, rank * 16 + 1),
            slice(0, 8),
        )
        assert lm_head_tiles[0].destination_slices == (
            slice(0, 1),
            slice(0, 8),
        )


def test_tile_planner_rejects_malformed_contracts():
    _, _, _, binding_plan = assignment_helper._fixture(0, 1)

    cases = (
        (object(), 24, "exact Qwen35CheckpointBindingPlan"),
        (binding_plan, True, "max_tile_bytes"),
        (binding_plan, 0, "max_tile_bytes"),
        (binding_plan, 23, "indivisible tile unit"),
    )
    for plan, budget, message in cases:
        _expect_error(
            lambda plan=plan, budget=budget: (
                build_qwen35_checkpoint_tile_plan(
                    plan,
                    max_tile_bytes=budget,
                )
            ),
            message,
        )

    first = binding_plan.bindings[0]
    malformed_plans = (
        (
            replace(
                binding_plan,
                bindings=(object(), *binding_plan.bindings[1:]),
            ),
            "exact Qwen35CheckpointTensorBinding",
        ),
        (
            replace(
                binding_plan,
                bindings=(
                    replace(first, loader_kind="unknown"),
                    *binding_plan.bindings[1:],
                ),
            ),
            "unsupported tiled checkpoint binding",
        ),
        (
            replace(
                binding_plan,
                bindings=(
                    replace(
                        first,
                        load=replace(
                            first.load,
                            transform="unknown",
                        ),
                    ),
                    *binding_plan.bindings[1:],
                ),
            ),
            "unsupported checkpoint transform",
        ),
        (
            replace(
                binding_plan,
                bindings=(
                    replace(
                        first,
                        load=replace(
                            first.load,
                            weight=replace(
                                first.load.weight,
                                target="layers.0.unknown.weight",
                            ),
                        ),
                    ),
                    *binding_plan.bindings[1:],
                ),
            ),
            "unsupported tiled checkpoint binding",
        ),
    )

    packed = next(
        binding
        for binding in binding_plan.bindings
        if binding.destination_slice is not None
    )
    packed_index = binding_plan.bindings.index(packed)
    malformed_plans += ((
        replace(
            binding_plan,
            bindings=(
                *binding_plan.bindings[:packed_index],
                replace(packed, destination_slice=(0, 999)),
                *binding_plan.bindings[packed_index + 1:],
            ),
        ),
        "destination slice",
    ),)

    second = binding_plan.bindings[1]
    conflicting = replace(
        first,
        load=replace(
            first.load,
            weight=replace(
                first.load.weight,
                source=replace(
                    first.load.weight.source,
                    name=second.load.weight.source.name,
                ),
            ),
        ),
    )
    malformed_plans += ((
        replace(
            binding_plan,
            bindings=(
                conflicting,
                *binding_plan.bindings[1:],
            ),
        ),
        "conflicting checkpoint source contract",
    ),)

    wrong_global_rows = replace(
        first,
        load=replace(
            first.load,
            metadata=replace(
                first.load.metadata,
                shape=(
                    first.load.metadata.shape[0] - 1,
                    *first.load.metadata.shape[1:],
                ),
            ),
        ),
    )
    malformed_plans += ((
        replace(
            binding_plan,
            bindings=(
                wrong_global_rows,
                *binding_plan.bindings[1:],
            ),
        ),
        "global rows must match TP-local rows",
    ),)

    wrong_offsets = replace(
        first,
        load=replace(
            first.load,
            metadata=replace(
                first.load.metadata,
                data_offsets=(
                    first.load.metadata.data_offsets[0],
                    first.load.metadata.data_offsets[1] - 2,
                ),
            ),
        ),
    )
    malformed_plans += ((
        replace(
            binding_plan,
            bindings=(
                wrong_offsets,
                *binding_plan.bindings[1:],
            ),
        ),
        "metadata byte count",
    ),)

    row_binding = next(
        binding
        for binding in binding_plan.bindings
        if binding.load.weight.target.endswith(
            "mlp.down_proj.weight"
        )
    )
    row_index = binding_plan.bindings.index(row_binding)
    wrong_global_columns = replace(
        row_binding,
        load=replace(
            row_binding.load,
            metadata=replace(
                row_binding.load.metadata,
                shape=(
                    row_binding.load.metadata.shape[0],
                    row_binding.load.metadata.shape[1] - 1,
                ),
            ),
        ),
    )
    malformed_plans += ((
        replace(
            binding_plan,
            bindings=(
                *binding_plan.bindings[:row_index],
                wrong_global_columns,
                *binding_plan.bindings[row_index + 1:],
            ),
        ),
        "replicated shape is invalid",
    ),)

    for plan, message in malformed_plans:
        _expect_error(
            lambda plan=plan: build_qwen35_checkpoint_tile_plan(
                plan,
                max_tile_bytes=24,
            ),
            message,
        )

    qkv = next(
        binding
        for binding in binding_plan.bindings
        if binding.load.weight.target.endswith(
            "linear_attention.in_proj_qkv.weight"
        )
    )
    qkv_index = binding_plan.bindings.index(qkv)
    wrong_qkv_local_shape = replace(
        qkv,
        local_shape=(
            qkv.local_shape[0] - 1,
            qkv.local_shape[1],
        ),
    )
    _expect_error(
        lambda: build_qwen35_checkpoint_tile_plan(
            replace(
                binding_plan,
                bindings=(
                    *binding_plan.bindings[:qkv_index],
                    wrong_qkv_local_shape,
                    *binding_plan.bindings[qkv_index + 1:],
                ),
            ),
            max_tile_bytes=24,
        ),
        "destination shape is invalid",
    )


def main():
    test_plans_all_five_binding_classes_at_tp_1_and_2()
    test_representative_source_and_destination_slices()
    test_tile_planner_rejects_malformed_contracts()
    print("qwen35 checkpoint tile planner tests passed (3 tests)")


if __name__ == "__main__":
    main()
