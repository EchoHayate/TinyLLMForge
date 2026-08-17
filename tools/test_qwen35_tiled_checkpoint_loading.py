from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
import weakref

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


streaming_helper = _load_helper(
    "qwen35_streamed_checkpoint_test_helper_for_tiles",
    "tools/test_qwen35_streamed_fresh_checkpoint.py",
)
reader_helper = streaming_helper.reader_helper
assignment_helper = reader_helper.assignment_helper

from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
    Qwen35TiledCheckpointLoadStats,
    Qwen35TiledLoadedCheckpointCandidate,
    load_qwen35_fresh_checkpoint_candidate_tiled as _load_qwen35_fresh_checkpoint_candidate_tiled,
)
import tinyvllm.models.qwen35_checkpoint_tiled_loading as tiled_module
from tinyvllm.engine.qwen35_hybrid_model_publication import (
    Qwen35HybridModelOwnerPublicationSlot,
)
from tinyvllm.models.qwen35_checkpoint_streaming import (
    load_qwen35_fresh_checkpoint_candidate as _load_qwen35_fresh_checkpoint_candidate,
)

MODEL_FINGERPRINT = "a" * 64


def load_qwen35_fresh_checkpoint_candidate_tiled(*args, **kwargs):
    kwargs.setdefault("model_fingerprint", MODEL_FINGERPRINT)
    return _load_qwen35_fresh_checkpoint_candidate_tiled(*args, **kwargs)


def load_qwen35_fresh_checkpoint_candidate(*args, **kwargs):
    kwargs.setdefault("model_fingerprint", MODEL_FINGERPRINT)
    return _load_qwen35_fresh_checkpoint_candidate(*args, **kwargs)


class _FailingLoader:

    def __init__(self, owner):
        self.__self__ = owner

    def __call__(self, *_):
        raise AssertionError("full-source custom loader must not execute")


def _factory(rank, world_size, diagnostics, *, failing_loaders=False):
    calls = []

    def build():
        calls.append(object())
        model, pool, tensor_plan, binding_plan, sources = (
            reader_helper._fixture(rank, world_size)
        )
        assignment_helper._initialize_destinations(binding_plan)
        if failing_loaders:
            for binding in binding_plan.bindings:
                loader = getattr(
                    binding.destination,
                    "weight_loader",
                    None,
                )
                if callable(loader):
                    binding.destination.weight_loader = _FailingLoader(
                        getattr(loader, "__self__", None)
                    )
        diagnostics.append((
            model,
            pool,
            tensor_plan,
            binding_plan,
            sources,
        ))
        return model, binding_plan

    return build, calls


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


class _SliceOnlyHandle:

    def __init__(self, handle, events, *, corrupt=None):
        self._handle = handle
        self._events = events
        self._corrupt = corrupt

    def keys(self):
        return self._handle.keys()

    def get_tensor(self, name):
        raise AssertionError(f"get_tensor must not execute: {name}")

    def get_slice(self, name):
        self._events.append(("slice", name))
        real = self._handle.get_slice(name)
        if self._corrupt is None:
            return real
        return _CorruptSlice(real, self._corrupt)


class _CorruptSlice:

    def __init__(self, real, corruption):
        self._real = real
        self._corruption = corruption

    def get_shape(self):
        shape = list(self._real.get_shape())
        if self._corruption == "metadata":
            shape[0] += 1
        return shape

    def __getitem__(self, item):
        tensor = self._real[item]
        if self._corruption == "dtype":
            return tensor.float()
        if self._corruption == "shape":
            return tensor[:-1]
        return tensor


def _tracking_safe_open(events, *, corrupt=None):
    real_safe_open = tiled_module.safe_open

    @contextmanager
    def tracked(*args, **kwargs):
        path = str(args[0])
        events.append(("enter", path))
        try:
            with real_safe_open(*args, **kwargs) as handle:
                yield _SliceOnlyHandle(
                    handle,
                    events,
                    corrupt=corrupt,
                )
        finally:
            events.append(("exit", path))

    return tracked


def _assert_balanced(events):
    entered = [value for action, value in events if action == "enter"]
    exited = [value for action, value in events if action == "exit"]
    assert entered == exited


def test_loads_exact_values_with_get_slice_only_and_no_custom_loaders():
    budgets = {1: 24, 2: 16}
    for world_size in (1, 2):
        for rank in range(world_size):
            diagnostics = []
            factory, calls = _factory(
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
                tiled_module.safe_open = _tracking_safe_open(events)
                try:
                    result = (
                        load_qwen35_fresh_checkpoint_candidate_tiled(
                            factory,
                            directory,
                            max_tile_bytes=budgets[world_size],
                        )
                    )
                finally:
                    tiled_module.safe_open = original

            assert len(calls) == 1
            model, pool, _, binding_plan, expected_sources = diagnostics[0]
            expected = assignment_helper._expected_destinations(
                binding_plan,
                expected_sources,
            )
            assert type(result) is Qwen35TiledLoadedCheckpointCandidate
            assert type(result.stats) is Qwen35TiledCheckpointLoadStats
            assert result.owner.model is model
            assert result.owner.pool is pool
            assert result.binding_plan is binding_plan
            assert result.tile_plan.binding_count == 27
            assert result.stats.assigned_bindings == 27
            assert result.stats.source_tensors == 27
            assert result.stats.shard_count == 2
            assert result.stats.tile_count == len(
                result.tile_plan.tiles
            )
            assert result.stats.destination_bytes == (
                result.tile_plan.destination_bytes
            )
            assert result.stats.materialized_bytes == (
                result.tile_plan.destination_bytes
            )
            assert result.stats.peak_tile_bytes <= budgets[world_size]
            assert result.model_fingerprint == MODEL_FINGERPRINT
            assert not hasattr(result, "tile_tensors")
            for object_id, destination in (
                assignment_helper._unique_destinations(
                    binding_plan
                ).items()
            ):
                torch.testing.assert_close(
                    destination,
                    expected[object_id],
                )
            assert (
                model.embed_tokens.weight.untyped_storage().data_ptr()
                == model.lm_head.weight.untyped_storage().data_ptr()
            )
            sliced_sources = [
                value
                for action, value in events
                if action == "slice"
            ]
            assert len(sliced_sources) == 27
            assert set(sliced_sources) == set(expected_sources)
            _assert_balanced(events)


def test_releases_each_materialized_tile_before_the_next_copy():
    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    previous = []
    original = tiled_module._copy_qwen35_checkpoint_tile

    def tracked(tile, tensor):
        gc.collect()
        if previous:
            assert previous[-1]() is None
        previous.append(weakref.ref(tensor))
        return original(tile, tensor)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        tiled_module._copy_qwen35_checkpoint_tile = tracked
        try:
            result = load_qwen35_fresh_checkpoint_candidate_tiled(
                factory,
                directory,
                max_tile_bytes=24,
            )
        finally:
            tiled_module._copy_qwen35_checkpoint_tile = original

    gc.collect()
    assert len(previous) == result.stats.tile_count
    assert previous[-1]() is None
    assert not hasattr(result, "tile_tensors")


def test_tiled_loader_rejects_slice_corruption_and_closes_handles():
    for corruption, message in (
        ("metadata", "source shape"),
        ("dtype", "dtype"),
        ("shape", "shape"),
    ):
        diagnostics = []
        factory, _ = _factory(0, 1, diagnostics)
        _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reader_helper._write_shards(
                directory,
                tensor_plan,
                sources,
            )
            events = []
            original = tiled_module.safe_open
            tiled_module.safe_open = _tracking_safe_open(
                events,
                corrupt=corruption,
            )
            try:
                _expect_error(
                    lambda: (
                        load_qwen35_fresh_checkpoint_candidate_tiled(
                            factory,
                            directory,
                            max_tile_bytes=24,
                        )
                    ),
                    message,
                )
            finally:
                tiled_module.safe_open = original
        _assert_balanced(events)

    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        original_copy = tiled_module._copy_qwen35_checkpoint_tile
        calls = []

        def failing_copy(tile, tensor):
            calls.append(tile)
            if len(calls) == 3:
                raise RuntimeError("injected tiled destination failure")
            return original_copy(tile, tensor)

        events = []
        original_open = tiled_module.safe_open
        tiled_module.safe_open = _tracking_safe_open(events)
        tiled_module._copy_qwen35_checkpoint_tile = failing_copy
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate_tiled(
                    factory,
                    directory,
                    max_tile_bytes=24,
                ),
                "injected tiled destination failure",
            )
        finally:
            tiled_module.safe_open = original_open
            tiled_module._copy_qwen35_checkpoint_tile = original_copy
    _assert_balanced(events)
    assert len(calls) == 3
    _, _, _, binding_plan, _ = diagnostics[0]
    assert not torch.all(binding_plan.bindings[0].destination == -7)

    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    first_load = tensor_plan.loads[0]
    first_name = first_load.weight.source.name
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        first_shard = directory / first_load.weight.source.shard
        remaining = {
            load.weight.source.name: sources[load.weight.source.name]
            for load in tensor_plan.loads
            if (
                load.weight.source.shard
                == first_load.weight.source.shard
                and load.weight.source.name != first_name
            )
        }
        from safetensors.torch import save_file
        save_file(remaining, first_shard)
        events = []
        original = tiled_module.safe_open
        tiled_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate_tiled(
                    factory,
                    directory,
                    max_tile_bytes=24,
                ),
                "missing requested source",
            )
        finally:
            tiled_module.safe_open = original
    _assert_balanced(events)


def test_tiled_loader_preopen_failures_do_not_open_shards():
    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, _, _, sources = reader_helper._fixture(0, 1)
    cases = (
        (object(), ".", 24, "candidate_factory"),
        (factory, ".", True, "max_tile_bytes"),
        (factory, ".", 23, "indivisible tile unit"),
        (factory, ".", 24, "SHA256"),
    )
    for candidate_factory, directory, budget, message in cases:
        events = []
        original = tiled_module.safe_open
        tiled_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda candidate_factory=candidate_factory,
                directory=directory,
                budget=budget: (
                    (
                        _load_qwen35_fresh_checkpoint_candidate_tiled(
                            candidate_factory,
                            directory,
                            max_tile_bytes=budget,
                            model_fingerprint="bad",
                        )
                        if message == "SHA256"
                        else load_qwen35_fresh_checkpoint_candidate_tiled(
                            candidate_factory,
                            directory,
                            max_tile_bytes=budget,
                        )
                    )
                ),
                message,
            )
        finally:
            tiled_module.safe_open = original
        assert events == []

    with tempfile.TemporaryDirectory() as temporary:
        events = []
        original = tiled_module.safe_open
        tiled_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate_tiled(
                    factory,
                    temporary,
                    max_tile_bytes=24,
                ),
                "missing checkpoint shard",
            )
        finally:
            tiled_module.safe_open = original
        assert events == []


def test_tiled_copy_casts_only_linear_attention_norm_weight():
    destination = torch.zeros(4, dtype=torch.bfloat16)
    source = torch.tensor(
        [0.125, -0.25, 0.375, -0.5],
        dtype=torch.float32,
    )
    tile = tiled_module.Qwen35CheckpointTile(
        binding_index=0,
        source_name="norm",
        shard="model.safetensors",
        source_tensor_shape=(4,),
        source_slices=(slice(0, 4),),
        tile_shape=(4,),
        destination=destination,
        destination_slices=(slice(0, 4),),
        destination_shape=(4,),
        dtype=torch.float32,
        byte_count=16,
        target="layers.0.linear_attention.norm_weight",
        kind="replicated",
    )

    tiled_module._copy_qwen35_checkpoint_tile(tile, source)

    torch.testing.assert_close(
        destination,
        source.to(torch.bfloat16),
        rtol=0.0,
        atol=0.0,
    )
    invalid = replace(
        tile,
        target="layers.0.input_layernorm.weight",
    )
    _expect_error(
        lambda: tiled_module._copy_qwen35_checkpoint_tile(
            invalid,
            source,
        ),
        "destination tile dtype mismatch",
    )


def test_failed_tiled_load_preserves_occupied_publication_slot():
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        published_factory, _ = _factory(0, 1, [])
        published_candidate = load_qwen35_fresh_checkpoint_candidate(
            published_factory,
            directory,
            max_tensor_bytes=streaming_helper._expected_peak_bytes(
                sources
            ),
        )
        slot = Qwen35HybridModelOwnerPublicationSlot()
        slot.publish(published_candidate)
        original_owner = slot.owner

        failing_factory, _ = _factory(0, 1, [])
        original_copy = tiled_module._copy_qwen35_checkpoint_tile
        calls = []

        def failing_copy(tile, tensor):
            calls.append(tile)
            if len(calls) == 2:
                raise RuntimeError("injected tiled publication isolation")
            return original_copy(tile, tensor)

        tiled_module._copy_qwen35_checkpoint_tile = failing_copy
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate_tiled(
                    failing_factory,
                    directory,
                    max_tile_bytes=24,
                ),
                "injected tiled publication isolation",
            )
        finally:
            tiled_module._copy_qwen35_checkpoint_tile = original_copy

    assert slot.owner is original_owner


def main():
    test_loads_exact_values_with_get_slice_only_and_no_custom_loaders()
    test_releases_each_materialized_tile_before_the_next_copy()
    test_tiled_loader_rejects_slice_corruption_and_closes_handles()
    test_tiled_loader_preopen_failures_do_not_open_shards()
    test_tiled_copy_casts_only_linear_attention_norm_weight()
    test_failed_tiled_load_preserves_occupied_publication_slot()
    print("qwen35 tiled checkpoint loading tests passed (6 tests)")


if __name__ == "__main__":
    main()
