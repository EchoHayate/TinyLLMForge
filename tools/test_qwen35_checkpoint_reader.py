from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch
from safetensors.torch import save_file

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
    "qwen35_checkpoint_assignment_helper",
    "tools/test_qwen35_checkpoint_assignment.py",
)
reader_module = _load_helper(
    "tinyvllm.models.qwen35_checkpoint_reader",
    "tinyvllm/models/qwen35_checkpoint_reader.py",
)

Qwen35CheckpointMaterialization = (
    reader_module.Qwen35CheckpointMaterialization
)
Qwen35CheckpointLoadResult = reader_module.Qwen35CheckpointLoadResult
materialize_qwen35_checkpoint_sources = (
    reader_module.materialize_qwen35_checkpoint_sources
)
load_and_assign_qwen35_checkpoint = (
    reader_module.load_and_assign_qwen35_checkpoint
)


def _rewrite_plan_shards(tensor_plan):
    loads = []
    for index, load in enumerate(tensor_plan.loads):
        source = replace(
            load.weight.source,
            shard=f"model-{index % 2 + 1:05d}-of-00002.safetensors",
        )
        loads.append(replace(
            load,
            weight=replace(load.weight, source=source),
        ))
    return replace(tensor_plan, loads=tuple(loads))


def _fixture(rank, world_size):
    model, pool = assignment_helper.helper._fixture(rank, world_size)
    tensor_plan = _rewrite_plan_shards(
        assignment_helper.helper._tensor_plan()
    )
    binding_plan = (
        assignment_helper.helper.build_qwen35_checkpoint_binding_plan(
            model,
            tensor_plan,
            tensor_parallel_size=world_size,
            tensor_parallel_rank=rank,
        )
    )
    sources = assignment_helper._source_tensors(tensor_plan)
    return model, pool, tensor_plan, binding_plan, sources


def _write_shards(directory, tensor_plan, sources, *, overrides=None):
    overrides = {} if overrides is None else overrides
    by_shard = {}
    for load in tensor_plan.loads:
        source_name = load.weight.source.name
        by_shard.setdefault(load.weight.source.shard, {})[source_name] = (
            overrides.get(source_name, sources[source_name])
        )
    for shard_name, tensors in by_shard.items():
        tensors = dict(tensors)
        tensors[f"extra.{shard_name}"] = torch.arange(3)
        save_file(tensors, directory / shard_name)


def _required_bytes(tensor_plan):
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in assignment_helper._source_tensors(
            tensor_plan
        ).values()
    )


def _expect_error(function, message):
    try:
        function()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _tracking_safe_open(events):
    real_safe_open = reader_module.safe_open

    @contextmanager
    def tracked(*args, **kwargs):
        path = str(args[0])
        events.append(("enter", path))
        try:
            with real_safe_open(*args, **kwargs) as handle:
                yield handle
        finally:
            events.append(("exit", path))

    return tracked


def _assert_balanced(events):
    entered = [path for action, path in events if action == "enter"]
    exited = [path for action, path in events if action == "exit"]
    assert entered == exited


def test_materializes_only_requested_sources_and_closes_handles():
    _, _, tensor_plan, binding_plan, sources = _fixture(0, 1)
    required_bytes = _required_bytes(tensor_plan)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _write_shards(directory, tensor_plan, sources)
        events = []
        original = reader_module.safe_open
        reader_module.safe_open = _tracking_safe_open(events)
        try:
            result = materialize_qwen35_checkpoint_sources(
                binding_plan,
                directory,
                max_materialized_bytes=required_bytes,
            )
        finally:
            reader_module.safe_open = original

    assert type(result) is Qwen35CheckpointMaterialization
    assert result.source_count == 27
    assert result.shard_count == 2
    assert result.materialized_bytes == required_bytes
    assert set(result.source_tensors) == set(sources)
    for name, tensor in result.source_tensors.items():
        torch.testing.assert_close(tensor, sources[name])
        assert tensor.device.type == "cpu"
    _expect_error(
        lambda: result.source_tensors.__setitem__("x", torch.ones(1)),
        "__setitem__",
    )
    _assert_balanced(events)


def test_load_and_assign_tp_1_and_2_after_handles_close():
    for world_size in (1, 2):
        for rank in range(world_size):
            model, _, tensor_plan, binding_plan, sources = _fixture(
                rank,
                world_size,
            )
            assignment_helper._initialize_destinations(binding_plan)
            expected = assignment_helper._expected_destinations(
                binding_plan,
                sources,
            )
            required_bytes = _required_bytes(tensor_plan)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _write_shards(directory, tensor_plan, sources)
                events = []
                original = reader_module.safe_open
                reader_module.safe_open = _tracking_safe_open(events)
                try:
                    result = load_and_assign_qwen35_checkpoint(
                        binding_plan,
                        directory,
                        max_materialized_bytes=required_bytes,
                    )
                finally:
                    reader_module.safe_open = original

            assert type(result) is Qwen35CheckpointLoadResult
            assert result.assignment.assigned_bindings == 27
            assert result.materialization.source_count == 27
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
            _assert_balanced(events)


def test_budget_path_and_metadata_fail_before_assignment():
    _, _, tensor_plan, binding_plan, sources = _fixture(0, 1)
    assignment_helper._initialize_destinations(binding_plan)
    required_bytes = _required_bytes(tensor_plan)

    for budget in (True, 0, -1):
        snapshot = assignment_helper._snapshot_destinations(binding_plan)
        _expect_error(
            lambda budget=budget: materialize_qwen35_checkpoint_sources(
                binding_plan,
                ".",
                max_materialized_bytes=budget,
            ),
            "max_materialized_bytes",
        )
        assignment_helper._assert_destinations_equal(snapshot)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        events = []
        original = reader_module.safe_open
        reader_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_and_assign_qwen35_checkpoint(
                    binding_plan,
                    directory,
                    max_materialized_bytes=required_bytes - 1,
                ),
                "exceeds max_materialized_bytes",
            )
        finally:
            reader_module.safe_open = original
        assert events == []

    _expect_error(
        lambda: materialize_qwen35_checkpoint_sources(
            binding_plan,
            "/definitely/missing/qwen35",
            max_materialized_bytes=required_bytes,
        ),
        "checkpoint_dir",
    )

    first_load = tensor_plan.loads[0]
    first_name = first_load.weight.source.name
    cases = (
        (
            {first_name: sources[first_name][:-1]},
            "shape",
        ),
        (
            {first_name: sources[first_name].float()},
            "dtype",
        ),
    )
    for overrides, message in cases:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_shards(
                directory,
                tensor_plan,
                sources,
                overrides=overrides,
            )
            snapshot = assignment_helper._snapshot_destinations(
                binding_plan
            )
            events = []
            original = reader_module.safe_open
            reader_module.safe_open = _tracking_safe_open(events)
            try:
                _expect_error(
                    lambda: load_and_assign_qwen35_checkpoint(
                        binding_plan,
                        directory,
                        max_materialized_bytes=required_bytes,
                    ),
                    message,
                )
            finally:
                reader_module.safe_open = original
            assignment_helper._assert_destinations_equal(snapshot)
            _assert_balanced(events)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _write_shards(directory, tensor_plan, sources)
        first_shard = directory / first_load.weight.source.shard
        tensors = {
            load.weight.source.name: sources[load.weight.source.name]
            for load in tensor_plan.loads
            if (
                load.weight.source.shard
                == first_load.weight.source.shard
                and load.weight.source.name != first_name
            )
        }
        save_file(tensors, first_shard)
        events = []
        original = reader_module.safe_open
        reader_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: materialize_qwen35_checkpoint_sources(
                    binding_plan,
                    directory,
                    max_materialized_bytes=required_bytes,
                ),
                "missing requested source",
            )
        finally:
            reader_module.safe_open = original
        _assert_balanced(events)


def test_assignment_begins_only_after_all_handles_close_and_rolls_back():
    _, _, tensor_plan, binding_plan, sources = _fixture(0, 2)
    assignment_helper._initialize_destinations(binding_plan)
    snapshot = assignment_helper._snapshot_destinations(binding_plan)
    required_bytes = _required_bytes(tensor_plan)
    failing_binding = next(
        binding
        for binding in reversed(binding_plan.bindings)
        if binding.loader_kind == "custom_parameter_loader"
    )
    original_loader = failing_binding.destination.weight_loader
    events = []

    def failing_loader(*_):
        _assert_balanced(events)
        raise RuntimeError("injected reader assignment failure")

    failing_binding.destination.weight_loader = failing_loader
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _write_shards(directory, tensor_plan, sources)
        original = reader_module.safe_open
        reader_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_and_assign_qwen35_checkpoint(
                    binding_plan,
                    directory,
                    max_materialized_bytes=required_bytes,
                ),
                failing_binding.load.weight.source.name,
            )
        finally:
            reader_module.safe_open = original
            failing_binding.destination.weight_loader = original_loader

    _assert_balanced(events)
    assignment_helper._assert_destinations_equal(snapshot)


def main():
    test_materializes_only_requested_sources_and_closes_handles()
    test_load_and_assign_tp_1_and_2_after_handles_close()
    test_budget_path_and_metadata_fail_before_assignment()
    test_assignment_begins_only_after_all_handles_close_and_rolls_back()
    print("qwen35 checkpoint reader tests passed (4 tests)")


if __name__ == "__main__":
    main()
