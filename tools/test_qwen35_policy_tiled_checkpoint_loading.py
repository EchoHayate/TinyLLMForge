from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
import weakref

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


tiled_helper = _load_helper(
    "qwen35_tiled_loading_policy_helper",
    "tools/test_qwen35_tiled_checkpoint_loading.py",
)
reader_helper = tiled_helper.reader_helper
assignment_helper = tiled_helper.assignment_helper

from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
    Qwen35PolicyTiledLoadedCheckpointCandidate,
    load_qwen35_fresh_checkpoint_candidate_with_tile_policy as _load_qwen35_fresh_checkpoint_candidate_with_tile_policy,
)
import tinyvllm.models.qwen35_checkpoint_tiled_loading as tiled_module

MODEL_FINGERPRINT = "a" * 64


def load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
    *args,
    **kwargs,
):
    kwargs.setdefault("model_fingerprint", MODEL_FINGERPRINT)
    return _load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
        *args,
        **kwargs,
    )


def _expect_error(function, message):
    try:
        function()
    except (
        AssertionError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_policy_loads_exact_candidate_once_at_tp_1_and_2():
    for world_size in (1, 2):
        for rank in range(world_size):
            diagnostics = []
            factory, calls = tiled_helper._factory(
                rank,
                world_size,
                diagnostics,
                failing_loaders=True,
            )
            _, _, tensor_plan, _, sources = reader_helper._fixture(
                rank,
                world_size,
            )
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reader_helper._write_shards(
                    directory,
                    tensor_plan,
                    sources,
                )
                events = []
                original = tiled_module.safe_open
                tiled_module.safe_open = (
                    tiled_helper._tracking_safe_open(events)
                )
                try:
                    result = (
                        load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
                            factory,
                            directory,
                            max_tile_bytes=256,
                            max_tile_count=40,
                        )
                    )
                finally:
                    tiled_module.safe_open = original

            assert len(calls) == 1
            assert type(result) is (
                Qwen35PolicyTiledLoadedCheckpointCandidate
            )
            assert result.loaded.tile_plan is result.decision.tile_plan
            assert result.loaded.stats.tile_count <= 40
            assert result.loaded.model_fingerprint == MODEL_FINGERPRINT
            assert result.loaded.stats.tile_count == len(
                result.decision.tile_plan.tiles
            )
            expected_budget = 128 if world_size == 1 else 64
            assert (
                result.decision.selected_max_tile_bytes
                == expected_budget
            )
            model, pool, _, binding_plan, expected_sources = diagnostics[0]
            assert result.loaded.owner.model is model
            assert result.loaded.owner.pool is pool
            expected = assignment_helper._expected_destinations(
                binding_plan,
                expected_sources,
            )
            for object_id, destination in (
                assignment_helper._unique_destinations(
                    binding_plan
                ).items()
            ):
                assignment_helper.torch.testing.assert_close(
                    destination,
                    expected[object_id],
                )
            sliced = [
                value
                for action, value in events
                if action == "slice"
            ]
            assert len(sliced) == 27
            assert set(sliced) == set(expected_sources)
            tiled_helper._assert_balanced(events)


def test_policy_loader_releases_tiles_and_fails_before_open():
    diagnostics = []
    factory, _ = tiled_helper._factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    previous = []
    original_copy = tiled_module._copy_qwen35_checkpoint_tile

    def tracked_copy(tile, tensor):
        gc.collect()
        if previous:
            assert previous[-1]() is None
        previous.append(weakref.ref(tensor))
        return original_copy(tile, tensor)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        tiled_module._copy_qwen35_checkpoint_tile = tracked_copy
        try:
            result = (
                load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
                    factory,
                    directory,
                    max_tile_bytes=256,
                    max_tile_count=40,
                )
            )
        finally:
            tiled_module._copy_qwen35_checkpoint_tile = original_copy
    gc.collect()
    assert len(previous) == result.loaded.stats.tile_count
    assert previous[-1]() is None

    events = []
    original_open = tiled_module.safe_open
    tiled_module.safe_open = tiled_helper._tracking_safe_open(events)
    try:
        _expect_error(
            lambda: (
                load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
                    tiled_helper._factory(0, 1, [])[0],
                    ".",
                    max_tile_bytes=64,
                    max_tile_count=40,
                )
            ),
            "cannot satisfy max_tile_count",
        )
    finally:
        tiled_module.safe_open = original_open
    assert events == []


def test_policy_load_failure_closes_handles_and_discards_candidate():
    diagnostics = []
    factory, calls = tiled_helper._factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        copy_calls = []
        original_copy = tiled_module._copy_qwen35_checkpoint_tile

        def failing_copy(tile, tensor):
            copy_calls.append(tile)
            if len(copy_calls) == 4:
                raise RuntimeError("injected policy tiled load failure")
            return original_copy(tile, tensor)

        events = []
        original_open = tiled_module.safe_open
        tiled_module.safe_open = tiled_helper._tracking_safe_open(events)
        tiled_module._copy_qwen35_checkpoint_tile = failing_copy
        try:
            _expect_error(
                lambda: (
                    load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
                        factory,
                        directory,
                        max_tile_bytes=256,
                        max_tile_count=40,
                    )
                ),
                "injected policy tiled load failure",
            )
        finally:
            tiled_module.safe_open = original_open
            tiled_module._copy_qwen35_checkpoint_tile = original_copy

    assert len(calls) == 1
    assert len(copy_calls) == 4
    tiled_helper._assert_balanced(events)
    _, _, _, binding_plan, _ = diagnostics[0]
    assert not assignment_helper.torch.all(
        binding_plan.bindings[0].destination == -7
    )


def main():
    test_policy_loads_exact_candidate_once_at_tp_1_and_2()
    test_policy_loader_releases_tiles_and_fails_before_open()
    test_policy_load_failure_closes_handles_and_discards_candidate()
    print("qwen35 policy tiled checkpoint loading tests passed (3 tests)")


if __name__ == "__main__":
    main()
