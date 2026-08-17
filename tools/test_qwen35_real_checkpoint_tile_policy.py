from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys

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
    "qwen35_real_checkpoint_tile_policy_helper",
    "tools/test_qwen35_real_checkpoint_tiles.py",
)

from tinyvllm.models.qwen35_checkpoint_tile_policy import (
    select_qwen35_checkpoint_tile_budget,
)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_real_policy_selects_smallest_budget_under_512_tiles():
    config, index_payload, shard_headers = (
        real_helper.real_helper._load_metadata()
    )
    tensor_plan = (
        real_helper.real_helper.build_qwen35_checkpoint_tensor_plan(
            config,
            index_payload,
            shard_headers,
        )
    )
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            raise AssertionError("safetensors payload must not be opened")
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        for world_size in (1, 2):
            for rank in range(world_size):
                binding_plan = real_helper._build_binding_plan(
                    config,
                    tensor_plan,
                    world_size,
                    rank,
                )
                result = select_qwen35_checkpoint_tile_budget(
                    binding_plan,
                    max_tile_bytes=16 << 20,
                    max_tile_count=512,
                )
                expected_budget = (
                    16 << 20 if world_size == 1 else 8 << 20
                )
                assert result.selected_max_tile_bytes == expected_budget
                assert len(result.tile_plan.tiles) == 488
                assert result.evaluations[-1].tile_count == 488

                if world_size == 1:
                    assert (
                        result.evaluations[-2].max_tile_bytes
                        == 8 << 20
                    )
                    assert result.evaluations[-2].tile_count == 651
                    _expect_error(
                        lambda binding_plan=binding_plan: (
                            select_qwen35_checkpoint_tile_budget(
                                binding_plan,
                                max_tile_bytes=8 << 20,
                                max_tile_count=512,
                            )
                        ),
                        "cannot satisfy max_tile_count",
                    )
                else:
                    assert (
                        result.evaluations[-2].max_tile_bytes
                        == 4 << 20
                    )
                    assert result.evaluations[-2].tile_count == 651
                    smaller = select_qwen35_checkpoint_tile_budget(
                        binding_plan,
                        max_tile_bytes=8 << 20,
                        max_tile_count=512,
                    )
                    assert smaller.selected_max_tile_bytes == 8 << 20
                    assert len(smaller.tile_plan.tiles) == 488
    finally:
        builtins.open = original_open


def main():
    test_real_policy_selects_smallest_budget_under_512_tiles()
    print("qwen35 real checkpoint tile policy tests passed (1 test)")


if __name__ == "__main__":
    main()
