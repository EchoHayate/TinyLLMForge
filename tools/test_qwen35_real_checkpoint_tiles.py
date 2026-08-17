from __future__ import annotations

import builtins
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


real_helper = _load_helper(
    "qwen35_real_component_binding_test_helper",
    "tools/test_qwen35_real_component_binding.py",
)

from tinyvllm.models.qwen35_checkpoint_tiles import (
    build_qwen35_checkpoint_tile_plan,
)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _binding_kinds(tile_plan):
    by_binding = {}
    for tile in tile_plan.tiles:
        existing = by_binding.setdefault(
            tile.binding_index,
            tile.kind,
        )
        assert existing == tile.kind
    return tuple(by_binding[index] for index in sorted(by_binding))


def _expected_local_bytes(binding_plan):
    byte_widths = {
        "BF16": 2,
        "F32": 4,
    }
    total = 0
    for binding in binding_plan.bindings:
        elements = 1
        for dimension in binding.local_shape:
            elements *= dimension
        total += elements * byte_widths[binding.load.metadata.dtype]
    return total


def _build_binding_plan(config, tensor_plan, world_size, rank):
    layout = real_helper.build_qwen35_hybrid_state_layout(
        config,
        tensor_parallel_size=world_size,
        dtype=torch.bfloat16,
    )
    pool = real_helper.HybridStateTensorPool(
        layout,
        capacity=1,
        device="cpu",
    )

    def build_backend(*arguments):
        return real_helper._StaticAttentionBackend(*arguments)

    target = real_helper.prepare_qwen35_checkpoint_candidate_target(
        config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
        build_attention_backend=build_backend,
        parameter_device="meta",
    )
    return target.binding_plan


def test_real_320_entry_tile_plan_is_bounded_and_complete():
    config, index_payload, shard_headers = real_helper._load_metadata()
    tensor_plan = real_helper.build_qwen35_checkpoint_tensor_plan(
        config,
        index_payload,
        shard_headers,
    )
    assert len(tensor_plan.loads) == 320
    budgets = {1: 12_288, 2: 12_288}
    expected_counts = {
        "axis0": 55,
        "axis1": 6,
        "segmented_axis0": 18,
        "squeeze_axis0": 18,
        "replicated": 223,
    }

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            raise AssertionError("safetensors payload must not be opened")
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        for world_size in (1, 2):
            for rank in range(world_size):
                binding_plan = _build_binding_plan(
                    config,
                    tensor_plan,
                    world_size,
                    rank,
                )
                result = build_qwen35_checkpoint_tile_plan(
                    binding_plan,
                    max_tile_bytes=budgets[world_size],
                )
                assert result.binding_count == 320
                assert result.source_count == 320
                assert result.tensor_parallel_size == world_size
                assert result.tensor_parallel_rank == rank
                assert result.peak_tile_bytes <= budgets[world_size]
                assert result.destination_bytes == (
                    _expected_local_bytes(binding_plan)
                )
                kinds = _binding_kinds(result)
                assert {
                    kind: kinds.count(kind)
                    for kind in set(kinds)
                } == expected_counts
                assert set(
                    tile.binding_index for tile in result.tiles
                ) == set(range(320))
                assert all(
                    0 < tile.byte_count <= budgets[world_size]
                    for tile in result.tiles
                )

                _expect_error(
                    lambda binding_plan=binding_plan,
                    world_size=world_size: (
                        build_qwen35_checkpoint_tile_plan(
                            binding_plan,
                            max_tile_bytes=budgets[world_size] - 1,
                        )
                    ),
                    "mlp.down_proj.weight",
                )
    finally:
        builtins.open = original_open


def main():
    test_real_320_entry_tile_plan_is_bounded_and_complete()
    print("qwen35 real checkpoint tile tests passed (1 test)")


if __name__ == "__main__":
    main()
