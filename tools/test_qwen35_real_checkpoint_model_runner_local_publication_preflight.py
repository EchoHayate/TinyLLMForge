from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/"
    "qwen35_real_checkpoint_model_runner_local_publication_preflight.py"
)
COMPLETE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)
PUBLICATION_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-private-publication-20260728-093000/"
    "private_publication_slot_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_model_runner_local_publication_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisites_source_closure_and_method_identity():
    module = _load_module()
    prerequisites = module.load_model_runner_publication_prerequisites(
        COMPLETE_ARTIFACT,
        PUBLICATION_ARTIFACT,
    )
    assert prerequisites.complete_artifact_sha256 == (
        "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
    )
    assert prerequisites.publication_artifact_sha256 == (
        "f208a799eca053e03a35aa4bfcbe66dfe6e5875b3e7b78390ded345a7c7c12b6"
    )
    assert prerequisites.publication_source_tree_sha256 == (
        "20c87258ff71449ebb8bf15af6ba77153804c16ab88a5fb11917a4597be51440"
    )
    assert tuple(prerequisites.complete_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert tuple(prerequisites.publication_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert len(prerequisites.publication_source_file_sha256) == 47
    assert len(module.SOURCE_FILES) == 49
    assert len(set(module.SOURCE_FILES)) == 49
    assert set(module.SOURCE_FILES) - set(
        prerequisites.publication_source_file_sha256
    ) == {
        "tinyvllm/engine/model_runner.py",
        "tools/qwen35_real_checkpoint_model_runner_local_publication_preflight.py",
    }
    method = module.load_frozen_model_runner_publication_method(ROOT)
    assert callable(method)
    assert module.MODEL_RUNNER_FILE_SHA256 == (
        "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
    )
    assert module.MODEL_RUNNER_PUBLICATION_METHOD_SHA256 == (
        "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
    )


def test_extracted_method_delegates_once_and_returns_exact_candidate():
    module = _load_module()
    method = module.load_frozen_model_runner_publication_method(ROOT)
    candidate = object()

    class Slot:
        def __init__(self, message=None):
            self.calls = []
            self.message = message

        def publish(self, value):
            self.calls.append(value)
            if self.message is not None:
                raise RuntimeError(self.message)

    slot = Slot()
    runner = SimpleNamespace(
        qwen35_loaded_checkpoint_candidate_slot=slot
    )
    assert method(runner, candidate) is candidate
    assert slot.calls == [candidate]

    rejecting = Slot("injected ModelRunner local publication failure")
    runner = SimpleNamespace(
        qwen35_loaded_checkpoint_candidate_slot=rejecting
    )
    try:
        method(runner, candidate)
    except RuntimeError as error:
        assert str(error) == (
            "injected ModelRunner local publication failure"
        )
    else:
        raise AssertionError("rejecting slot must fail")
    assert rejecting.calls == [candidate]


def test_publication_prerequisite_has_six_unique_processes():
    record = json.loads(PUBLICATION_ARTIFACT.read_text())
    assert len(record["rows"]) == 6
    assert len({row["process_id"] for row in record["rows"]}) == 6


def test_private_method_transaction_success_and_injected_failure():
    module = _load_module()
    method = module.load_frozen_model_runner_publication_method(ROOT)
    selected = torch.full((2,), 9, dtype=torch.bfloat16)
    expected = torch.tensor((1.0, 2.0), dtype=torch.bfloat16)

    class Value:
        pass

    class Model:
        def named_parameters(self, remove_duplicate=False):
            return (("selected", selected),)

        def named_buffers(self, remove_duplicate=False):
            return ()

    class Binding:
        destination = selected
        destination_slice = None

    class Slot:
        def __init__(self):
            self.candidate = None
            self.owner = None
            self.model_fingerprint = None
            self.calls = 0

        def publish(self, candidate):
            self.calls += 1
            self.candidate = candidate
            self.owner = candidate.owner
            self.model_fingerprint = candidate.model_fingerprint
            return candidate.owner

    oracle = {
        "binding_results": [{
            "binding_index": 0,
            "phase_name": "all",
            "destination_sha256": module._sha256(
                expected.view(torch.uint8).numpy().tobytes()
            ),
        }],
        "phase_results": [{
            "phase_name": "all",
            "destination_sha256": module._sha256(
                expected.view(torch.uint8).numpy().tobytes()
            ),
        }],
        "aggregate_destination_sha256": module._sha256(
            expected.view(torch.uint8).numpy().tobytes()
        ),
    }

    def private_graph_factory():
        model = Model()
        pool = Value()
        pool.marker = "pool"
        owner = Value()
        owner.model = model
        owner.pool = pool
        owner.layer_stack = Value()
        owner.state_transaction = Value()
        owner.runtime_bridge = Value()
        owner.runtime_bridge.pool = pool
        plan = Value()
        plan.bindings = (Binding(),)
        target = Value()
        target.assembly = Value()
        target.assembly.packed = Value()
        target.assembly.packed.model = model
        target.binding_plan = plan
        target.pool = pool
        target._consumed = False
        candidate = Value()
        candidate.owner = owner
        candidate.binding_plan = plan
        candidate.model_fingerprint = "a" * 64
        candidate.stats = SimpleNamespace(
            assigned_bindings=1,
            source_tensors=1,
            shard_count=1,
            loaded_bytes=4,
            peak_source_bytes=4,
        )

        def acquire():
            target._consumed = True
            selected.copy_(expected)
            return candidate

        return target, acquire

    result = module.execute_model_runner_local_publication_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=oracle,
        model_fingerprint="a" * 64,
        publication_method=method,
        production_slot_factory=Slot,
        mode="success",
    )
    assert result["method_call_count"] == 1
    assert result["production_publish_call_count"] == 1
    assert result["method_return_identity_verified"] is True
    assert result["production_slot_visibility_verified"] is True
    assert result["all_private_method_objects_collected"] is True
    assert not int(selected.count_nonzero().item())

    result = module.execute_model_runner_local_publication_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=oracle,
        model_fingerprint="a" * 64,
        publication_method=method,
        production_slot_factory=Slot,
        mode="injected_method_failure",
    )
    assert result["method_call_count"] == 1
    assert result["proxy_publish_call_count"] == 1
    assert result["production_publish_call_count"] == 0
    assert result["production_slot_remained_empty"] is True
    assert result["method_returned_candidate"] is False
    assert result["expected_failure_message"] == (
        "injected ModelRunner local publication failure"
    )
    assert result["all_private_method_objects_collected"] is True
    assert not int(selected.count_nonzero().item())


def test_real_worker_contract_source_closure_and_partial_rejection():
    module = _load_module()
    assert module.SCHEMA_VERSION == (
        "qwen35.real-checkpoint-model-runner-local-publication.v1"
    )
    assert module.ROW_SCHEMA_VERSION == (
        "qwen35.real-checkpoint-model-runner-local-publication-rank.v1"
    )
    assert module.WORKER_CONTEXTS == (
        (1, 0, "success"),
        (1, 0, "injected_method_failure"),
        (2, 0, "success"),
        (2, 0, "injected_method_failure"),
        (2, 1, "success"),
        (2, 1, "injected_method_failure"),
    )
    assert module.STREAMED_STATS == {
        (1, 0): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        (2, 0): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        (2, 1): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
    }
    hashes = module._source_hashes(ROOT)
    publication_record = json.loads(PUBLICATION_ARTIFACT.read_text())
    assert len(hashes) == 49
    assert {
        name: hashes[name]
        for name in publication_record["source_file_sha256"]
    } == publication_record["source_file_sha256"]
    assert hashes[module.MODEL_RUNNER_SOURCE] == (
        module.MODEL_RUNNER_FILE_SHA256
    )
    archive = module.build_source_tar(ROOT)
    assert isinstance(archive, bytes)
    assert len(archive) > 0
    try:
        module._aggregate([], ROOT)
    except ValueError as error:
        assert "worker rows" in str(error)
    else:
        raise AssertionError("partial finalization must fail")


def main():
    tests = (
        test_exact_prerequisites_source_closure_and_method_identity,
        test_extracted_method_delegates_once_and_returns_exact_candidate,
        test_publication_prerequisite_has_six_unique_processes,
        test_private_method_transaction_success_and_injected_failure,
        test_real_worker_contract_source_closure_and_partial_rejection,
    )
    for test in tests:
        test()
    print(
        "qwen35 ModelRunner local publication tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
