from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

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
    "qwen35_streamed_checkpoint_test_helper",
    "tools/test_qwen35_streamed_fresh_checkpoint.py",
)
reader_helper = streaming_helper.reader_helper

from tinyvllm.engine.qwen35_hybrid_model_publication import (
    Qwen35HybridModelOwnerPublicationSlot,
)
from tinyvllm.engine.hybrid_state import HybridStateRuntimeBridge
from tinyvllm.models.qwen35_checkpoint_streaming import (
    load_qwen35_fresh_checkpoint_candidate as _load_qwen35_fresh_checkpoint_candidate,
)

MODEL_FINGERPRINT = "a" * 64


def load_qwen35_fresh_checkpoint_candidate(*args, **kwargs):
    kwargs.setdefault("model_fingerprint", MODEL_FINGERPRINT)
    return _load_qwen35_fresh_checkpoint_candidate(*args, **kwargs)


def _loaded_candidate(rank=0, world_size=1):
    diagnostics = []
    factory, _ = streaming_helper._factory(
        rank,
        world_size,
        diagnostics,
    )
    _, _, tensor_plan, _, sources = reader_helper._fixture(
        rank,
        world_size,
    )
    temporary = tempfile.TemporaryDirectory()
    directory = Path(temporary.name)
    reader_helper._write_shards(
        directory,
        tensor_plan,
        sources,
    )
    candidate = load_qwen35_fresh_checkpoint_candidate(
        factory,
        directory,
        max_tensor_bytes=streaming_helper._expected_peak_bytes(
            sources
        ),
    )
    return temporary, candidate


def _expect_error(function, message):
    try:
        function()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_publication_slot_is_one_shot_and_preserves_identity():
    slot = Qwen35HybridModelOwnerPublicationSlot()
    assert slot.candidate is None
    assert slot.owner is None
    assert slot.model_fingerprint is None
    assert not hasattr(slot, "clear")
    assert not hasattr(slot, "replace")

    _expect_error(
        lambda: slot.publish(object()),
        "loaded checkpoint candidate",
    )
    assert slot.candidate is None
    assert slot.owner is None
    assert slot.model_fingerprint is None

    first_temporary, first = _loaded_candidate()
    second_temporary, second = _loaded_candidate()
    try:
        incoherent = replace(
            first,
            owner=replace(
                first.owner,
                runtime_bridge=HybridStateRuntimeBridge(
                    second.owner.pool
                ),
            ),
        )
        _expect_error(
            lambda: slot.publish(incoherent),
            "owner graph is incoherent",
        )
        assert slot.candidate is None
        assert slot.owner is None
        assert slot.model_fingerprint is None

        published = slot.publish(first)
        assert published is first.owner
        assert slot.candidate is first
        assert slot.owner is first.owner
        assert slot.model_fingerprint == MODEL_FINGERPRINT

        _expect_error(
            lambda: slot.publish(second),
            "already occupied",
        )
        assert slot.candidate is first
        assert slot.owner is first.owner
        assert slot.model_fingerprint == MODEL_FINGERPRINT
    finally:
        first_temporary.cleanup()
        second_temporary.cleanup()


def test_failed_stream_load_cannot_change_published_owner():
    temporary, candidate = _loaded_candidate()
    slot = Qwen35HybridModelOwnerPublicationSlot()
    slot.publish(candidate)
    original_owner = slot.owner

    diagnostics = []

    def inject_failure(model, plan, _tensor_plan, _sources):
        failing = next(
            binding
            for binding in reversed(plan.bindings)
            if binding.loader_kind == "custom_parameter_loader"
        )

        def failing_loader(*_):
            raise RuntimeError("injected publication isolation failure")

        failing.destination.weight_loader = failing_loader
        return model, plan

    factory, _ = streaming_helper._factory(
        0,
        1,
        diagnostics,
        mutate=inject_failure,
    )
    _, _, tensor_plan, _, sources = reader_helper._fixture(0, 1)
    try:
        with tempfile.TemporaryDirectory() as failed_temporary:
            directory = Path(failed_temporary)
            reader_helper._write_shards(
                directory,
                tensor_plan,
                sources,
            )
            _expect_error(
                lambda: load_qwen35_fresh_checkpoint_candidate(
                    factory,
                    directory,
                    max_tensor_bytes=(
                        streaming_helper._expected_peak_bytes(sources)
                    ),
                ),
                "injected publication isolation failure",
            )
        assert slot.owner is original_owner
    finally:
        temporary.cleanup()


def main():
    test_publication_slot_is_one_shot_and_preserves_identity()
    test_failed_stream_load_cannot_change_published_owner()
    print("qwen35 hybrid model publication tests passed (2 tests)")


if __name__ == "__main__":
    main()
