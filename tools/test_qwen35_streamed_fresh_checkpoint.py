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


reader_helper = _load_helper(
    "qwen35_checkpoint_reader_test_helper",
    "tools/test_qwen35_checkpoint_reader.py",
)
assignment_helper = reader_helper.assignment_helper

from tinyvllm.models.qwen35_checkpoint_streaming import (
    Qwen35LoadedCheckpointCandidate,
    Qwen35StreamedCheckpointLoadStats,
    load_qwen35_fresh_checkpoint_candidate as _load_qwen35_fresh_checkpoint_candidate,
)
import tinyvllm.models.qwen35_checkpoint_streaming as streaming_module

MODEL_FINGERPRINT = "a" * 64


def load_qwen35_fresh_checkpoint_candidate(*args, **kwargs):
    kwargs.setdefault("model_fingerprint", MODEL_FINGERPRINT)
    return _load_qwen35_fresh_checkpoint_candidate(*args, **kwargs)


def _factory(rank, world_size, diagnostics, *, mutate=None):
    calls = []

    def build():
        calls.append(object())
        model, pool, tensor_plan, binding_plan, sources = (
            reader_helper._fixture(rank, world_size)
        )
        assignment_helper._initialize_destinations(binding_plan)
        if mutate is not None:
            model, binding_plan = mutate(
                model,
                binding_plan,
                tensor_plan,
                sources,
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


def _expected_peak_bytes(sources):
    return max(
        tensor.numel() * tensor.element_size()
        for tensor in sources.values()
    )


def _expect_error(function, message):
    try:
        function()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _tracking_safe_open(events):
    real_safe_open = streaming_module.safe_open

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


def test_streams_fresh_candidate_exactly_at_tp_1_and_2():
    for world_size in (1, 2):
        for rank in range(world_size):
            diagnostics = []
            factory, calls = _factory(
                rank,
                world_size,
                diagnostics,
            )
            _, _, tensor_plan, _, sources = reader_helper._fixture(
                rank,
                world_size,
            )
            total_bytes = reader_helper._required_bytes(tensor_plan)
            peak_bytes = _expected_peak_bytes(sources)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reader_helper._write_shards(
                    directory,
                    tensor_plan,
                    sources,
                )
                events = []
                original = streaming_module.safe_open
                streaming_module.safe_open = _tracking_safe_open(events)
                try:
                    result = load_qwen35_fresh_checkpoint_candidate(
                        factory,
                        directory,
                        max_tensor_bytes=peak_bytes,
                    )
                finally:
                    streaming_module.safe_open = original

            assert len(calls) == 1
            assert len(diagnostics) == 1
            model, pool, _, binding_plan, expected_sources = diagnostics[0]
            expected = assignment_helper._expected_destinations(
                binding_plan,
                expected_sources,
            )
            assert type(result) is Qwen35LoadedCheckpointCandidate
            assert type(result.stats) is Qwen35StreamedCheckpointLoadStats
            assert result.owner.model is model
            assert result.owner.layer_stack is model.layer_stack
            assert result.owner.state_transaction is (
                model.layer_stack.state_transaction
            )
            assert result.owner.pool is pool
            assert result.binding_plan is binding_plan
            assert result.stats.assigned_bindings == 27
            assert result.stats.source_tensors == 27
            assert result.stats.shard_count == 2
            assert result.stats.loaded_bytes == total_bytes
            assert result.stats.peak_source_bytes == peak_bytes
            assert result.model_fingerprint == MODEL_FINGERPRINT
            assert not hasattr(result, "source_tensors")
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
            assert (
                model.embed_tokens.weight.storage_offset()
                == model.lm_head.weight.storage_offset()
            )
            _assert_balanced(events)


def test_releases_each_source_before_assigning_the_next():
    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    peak_bytes = _expected_peak_bytes(sources)
    previous = []
    original = (
        streaming_module._assign_qwen35_checkpoint_source_bindings
    )

    def tracked(bindings, source, **kwargs):
        gc.collect()
        if previous:
            assert previous[-1]() is None
        previous.append(weakref.ref(source))
        return original(bindings, source, **kwargs)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        streaming_module._assign_qwen35_checkpoint_source_bindings = (
            tracked
        )
        try:
            result = load_qwen35_fresh_checkpoint_candidate(
                factory,
                directory,
                max_tensor_bytes=peak_bytes,
            )
        finally:
            streaming_module._assign_qwen35_checkpoint_source_bindings = (
                original
            )

    gc.collect()
    assert len(previous) == 27
    assert previous[-1]() is None
    assert not hasattr(result, "source_tensors")


def test_rejects_invalid_candidate_contracts_before_open():
    diagnostics = []
    factory, _ = _factory(0, 1, diagnostics)
    _, _, tensor_plan, binding_plan, sources = reader_helper._fixture(0, 1)
    peak_bytes = _expected_peak_bytes(sources)

    cases = (
        (object(), ".", peak_bytes, "candidate_factory"),
        (lambda: object(), ".", peak_bytes, "two-item tuple"),
        (
            lambda: (object(), binding_plan),
            ".",
            peak_bytes,
            "exact Qwen35PackedForCausalLM",
        ),
        (
            lambda: (diagnostics, object()),
            ".",
            peak_bytes,
            "exact Qwen35PackedForCausalLM",
        ),
        (factory, ".", True, "max_tensor_bytes"),
        (factory, ".", 0, "max_tensor_bytes"),
        (factory, ".", peak_bytes, "SHA256"),
        (
            factory,
            ".",
            peak_bytes - 1,
            "exceeds max_tensor_bytes",
        ),
    )
    for candidate_factory, directory, budget, message in cases:
        events = []
        original = streaming_module.safe_open
        streaming_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda candidate_factory=candidate_factory,
                directory=directory,
                budget=budget: (
                    (
                        _load_qwen35_fresh_checkpoint_candidate(
                            candidate_factory,
                            directory,
                            max_tensor_bytes=budget,
                            model_fingerprint="bad",
                        )
                        if message == "SHA256"
                        else load_qwen35_fresh_checkpoint_candidate(
                            candidate_factory,
                            directory,
                            max_tensor_bytes=budget,
                        )
                    )
                ),
                message,
            )
        finally:
            streaming_module.safe_open = original
        assert events == []

    def meta_destination(model, plan, _tensor_plan, _sources):
        first = replace(
            plan.bindings[0],
            destination=torch.empty(
                plan.bindings[0].destination.shape,
                dtype=plan.bindings[0].destination.dtype,
                device="meta",
            ),
        )
        return model, replace(
            plan,
            bindings=(first, *plan.bindings[1:]),
        )

    def conflicting_source(model, plan, _tensor_plan, _sources):
        second = plan.bindings[1]
        first = replace(
            plan.bindings[0],
            load=replace(
                plan.bindings[0].load,
                weight=replace(
                    plan.bindings[0].load.weight,
                    source=replace(
                        plan.bindings[0].load.weight.source,
                        name=second.load.weight.source.name,
                    ),
                ),
            ),
        )
        return model, replace(
            plan,
            bindings=(first, *plan.bindings[1:]),
        )

    for mutate, message in (
        (meta_destination, "CPU non-meta"),
        (conflicting_source, "conflicting checkpoint source contract"),
    ):
        local_diagnostics = []
        bad_factory, _ = _factory(
            0,
            1,
            local_diagnostics,
            mutate=mutate,
        )
        events = []
        original = streaming_module.safe_open
        streaming_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate(
                    bad_factory,
                    ".",
                    max_tensor_bytes=peak_bytes,
                ),
                message,
            )
        finally:
            streaming_module.safe_open = original
        assert events == []

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        safe_factory, _ = _factory(0, 1, [])
        events = []
        original = streaming_module.safe_open
        streaming_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate(
                    safe_factory,
                    directory,
                    max_tensor_bytes=peak_bytes,
                ),
                "missing checkpoint shard",
            )
        finally:
            streaming_module.safe_open = original
        assert events == []


def test_stream_failures_close_handles_and_discard_without_rollback():
    for corruption, message in (("shape", "shape"), ("dtype", "dtype")):
        diagnostics = []
        factory, _ = _factory(0, 1, diagnostics)
        _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
        first_name = tensor_plan.loads[0].weight.source.name
        override = (
            sources[first_name][:-1]
            if corruption == "shape"
            else sources[first_name].float()
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reader_helper._write_shards(
                directory,
                tensor_plan,
                sources,
                overrides={first_name: override},
            )
            events = []
            original = streaming_module.safe_open
            streaming_module.safe_open = _tracking_safe_open(events)
            try:
                _expect_error(
                    lambda: load_qwen35_fresh_checkpoint_candidate(
                        factory,
                        directory,
                        max_tensor_bytes=_expected_peak_bytes(sources),
                    ),
                    message,
                )
            finally:
                streaming_module.safe_open = original
        _assert_balanced(events)

    diagnostics = []

    def inject_failure(model, plan, _tensor_plan, _sources):
        failing = next(
            binding
            for binding in reversed(plan.bindings)
            if binding.loader_kind == "custom_parameter_loader"
        )

        def failing_loader(*_):
            raise RuntimeError("injected streamed assignment failure")

        failing.destination.weight_loader = failing_loader
        return model, plan

    factory, _ = _factory(
        0,
        2,
        diagnostics,
        mutate=inject_failure,
    )
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 2)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reader_helper._write_shards(
            directory,
            tensor_plan,
            sources,
        )
        events = []
        original = streaming_module.safe_open
        streaming_module.safe_open = _tracking_safe_open(events)
        try:
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate(
                    factory,
                    directory,
                    max_tensor_bytes=_expected_peak_bytes(sources),
                ),
                "injected streamed assignment failure",
            )
        finally:
            streaming_module.safe_open = original

    _assert_balanced(events)
    _, _, _, binding_plan, _ = diagnostics[0]
    first_destination = binding_plan.bindings[0].destination
    assert not torch.all(first_destination == -7)


def main():
    test_streams_fresh_candidate_exactly_at_tp_1_and_2()
    test_releases_each_source_before_assigning_the_next()
    test_rejects_invalid_candidate_contracts_before_open()
    test_stream_failures_close_handles_and_discard_without_rollback()
    print("qwen35 streamed fresh checkpoint tests passed (4 tests)")


if __name__ == "__main__":
    main()
