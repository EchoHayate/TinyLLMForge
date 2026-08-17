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
    "qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py"
)
COMPLETE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)
PUBLICATION_METHOD_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-local-publication-20260728-090014/"
    "model_runner_local_publication_preflight.json"
)
MODEL_FINGERPRINT = "a" * 64
AUTHORIZATION_SHA256 = "b" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_model_runner_load_publish_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRequest:

    def __init__(
        self,
        checkpoint_dir="/approved/model",
        model_fingerprint=MODEL_FINGERPRINT,
        max_tensor_bytes=8 << 20,
        authorization_sha256=AUTHORIZATION_SHA256,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_fingerprint = model_fingerprint
        self.max_tensor_bytes = max_tensor_bytes
        self.authorization_sha256 = authorization_sha256

    def __eq__(self, other):
        return (
            type(other) is FakeRequest
            and self.checkpoint_dir == other.checkpoint_dir
            and self.model_fingerprint == other.model_fingerprint
            and self.max_tensor_bytes == other.max_tensor_bytes
            and self.authorization_sha256
            == other.authorization_sha256
        )


class FakeCandidate:

    def __init__(self, model_fingerprint=MODEL_FINGERPRINT):
        self.model_fingerprint = model_fingerprint


class FakeSlot:

    def __init__(self, message=None):
        self.candidate = None
        self.publish_calls = []
        self.message = message

    def publish(self, candidate):
        self.publish_calls.append(candidate)
        if self.message is not None:
            raise RuntimeError(self.message)
        if self.candidate is not None:
            raise RuntimeError("slot already occupied")
        self.candidate = candidate
        return candidate


def _validate_request(value):
    if type(value) is not FakeRequest:
        raise ValueError("request must be an exact request")
    return value


def _runner(slot, loader, rank=1):
    return SimpleNamespace(
        rank=rank,
        qwen35_checkpoint_candidate_loader=loader,
        qwen35_checkpoint_candidate_loader_authorization_sha256=(
            AUTHORIZATION_SHA256
        ),
        qwen35_checkpoint_candidate_load_configuration=None,
        qwen35_checkpoint_candidate_load_request=None,
        qwen35_loaded_checkpoint_candidate_slot=slot,
    )


def test_exact_prerequisites_source_closure_and_method_identity():
    module = _load_module()
    prerequisites = module.load_model_runner_load_publish_prerequisites(
        COMPLETE_ARTIFACT,
        PUBLICATION_METHOD_ARTIFACT,
    )
    assert prerequisites.complete_artifact_sha256 == (
        "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
    )
    assert prerequisites.publication_method_artifact_sha256 == (
        "f8f78ae574991eb3f16aed57b4275cf76a409fa553e01597f5179c41eb158b15"
    )
    assert prerequisites.publication_method_source_tree_sha256 == (
        "d3eb52326d8e9d9a744f4641877c90a41468d26f94cbe31eda5ee04fe4d2201a"
    )
    assert tuple(prerequisites.complete_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert tuple(prerequisites.publication_method_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert len(prerequisites.publication_method_source_file_sha256) == 49
    assert len(module.SOURCE_FILES) == 50
    assert len(set(module.SOURCE_FILES)) == 50
    assert set(module.SOURCE_FILES) - set(
        prerequisites.publication_method_source_file_sha256
    ) == {
        "tools/"
        "qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py"
    }
    method = module.load_frozen_model_runner_load_publish_method(
        ROOT,
        loaded_candidate_type=FakeCandidate,
        request_validator=_validate_request,
    )
    assert callable(method)
    assert module.MODEL_RUNNER_FILE_SHA256 == (
        "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
    )
    assert module.MODEL_RUNNER_LOAD_PUBLISH_METHOD_SHA256 == (
        "9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612"
    )


def test_extracted_method_success_commits_only_after_publication():
    module = _load_module()
    method = module.load_frozen_model_runner_load_publish_method(
        ROOT,
        loaded_candidate_type=FakeCandidate,
        request_validator=_validate_request,
    )
    request = FakeRequest()
    candidate = FakeCandidate()
    calls = []

    def loader(value):
        calls.append(value)
        return candidate

    slot = FakeSlot()
    runner = _runner(slot, loader, rank=2)
    row = method(runner, request)

    assert row == {
        "participant_id": 2,
        "operation": "load_checkpoint_candidate",
        "status": "published",
        "model_fingerprint": MODEL_FINGERPRINT,
        "detail": "",
    }
    assert calls == [request]
    assert slot.publish_calls == [candidate]
    assert slot.candidate is candidate
    assert runner.qwen35_checkpoint_candidate_load_request is request
    assert runner.qwen35_checkpoint_candidate_load_configuration == (
        "/approved/model",
        MODEL_FINGERPRINT,
        8 << 20,
        AUTHORIZATION_SHA256,
    )


def test_extracted_method_publication_failure_returns_error_and_no_commit():
    module = _load_module()
    method = module.load_frozen_model_runner_load_publish_method(
        ROOT,
        loaded_candidate_type=FakeCandidate,
        request_validator=_validate_request,
    )
    request = FakeRequest()
    candidate = FakeCandidate()
    calls = []

    def loader(value):
        calls.append(value)
        return candidate

    slot = FakeSlot("injected ModelRunner load-and-publish failure")
    runner = _runner(slot, loader, rank=3)
    row = method(runner, request)

    assert row == {
        "participant_id": 3,
        "operation": "load_checkpoint_candidate",
        "status": "error",
        "model_fingerprint": "",
        "detail": (
            "RuntimeError: "
            "injected ModelRunner load-and-publish failure"
        ),
    }
    assert calls == [request]
    assert slot.publish_calls == [candidate]
    assert slot.candidate is None
    assert runner.qwen35_checkpoint_candidate_load_request is None
    assert (
        runner.qwen35_checkpoint_candidate_load_configuration is None
    )


def test_publication_method_prerequisite_has_six_unique_processes():
    record = json.loads(PUBLICATION_METHOD_ARTIFACT.read_text())
    assert len(record["rows"]) == 6
    assert len({row["process_id"] for row in record["rows"]}) == 6


def test_private_load_publish_transaction_success_and_failure():
    module = _load_module()
    method = module.load_frozen_model_runner_load_publish_method(
        ROOT,
        loaded_candidate_type=FakeCandidate,
        request_validator=_validate_request,
    )
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
        candidate = FakeCandidate()
        candidate.owner = owner
        candidate.binding_plan = plan
        candidate.stats = SimpleNamespace(
            assigned_bindings=1,
            source_tensors=1,
            shard_count=1,
            loaded_bytes=4,
            peak_source_bytes=4,
        )
        request = FakeRequest()

        def loader(value):
            assert value is request
            target._consumed = True
            selected.copy_(expected)
            return candidate

        return target, request, loader

    for mode in ("success", "injected_publication_failure"):
        result = module.execute_model_runner_load_publish_scope(
            private_graph_factory=private_graph_factory,
            oracle_row=oracle,
            model_fingerprint=MODEL_FINGERPRINT,
            publication_method=method,
            production_slot_factory=FakeSlot,
            mode=mode,
            rank=2,
        )
        assert result["method_call_count"] == 1
        assert result["adapter_call_count"] == 1
        assert result["provider_call_count"] == 1
        assert result["all_private_load_publish_objects_collected"] is True
        assert not int(selected.count_nonzero().item())
        if mode == "success":
            assert result["method_row"]["status"] == "published"
            assert result["production_publish_call_count"] == 1
            assert result["completion_state_committed"] is True
        else:
            assert result["method_row"]["status"] == "error"
            assert result["method_row"]["detail"] == (
                "RuntimeError: "
                "injected ModelRunner load-and-publish failure"
            )
            assert result["proxy_publish_call_count"] == 1
            assert result["production_publish_call_count"] == 0
            assert result["completion_state_committed"] is False


def test_real_worker_contract_source_closure_and_partial_rejection():
    module = _load_module()
    assert module.SCHEMA_VERSION == (
        "qwen35.real-checkpoint-model-runner-load-and-publish.v1"
    )
    assert module.ROW_SCHEMA_VERSION == (
        "qwen35.real-checkpoint-model-runner-load-and-publish-rank.v1"
    )
    assert module.WORKER_CONTEXTS == (
        (1, 0, "success"),
        (1, 0, "injected_publication_failure"),
        (2, 0, "success"),
        (2, 0, "injected_publication_failure"),
        (2, 1, "success"),
        (2, 1, "injected_publication_failure"),
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
    publication_record = json.loads(
        PUBLICATION_METHOD_ARTIFACT.read_text()
    )
    assert len(hashes) == 50
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
        test_extracted_method_success_commits_only_after_publication,
        test_extracted_method_publication_failure_returns_error_and_no_commit,
        test_publication_method_prerequisite_has_six_unique_processes,
        test_private_load_publish_transaction_success_and_failure,
        test_real_worker_contract_source_closure_and_partial_rejection,
    )
    for test in tests:
        test()
    print(
        "qwen35 ModelRunner load-and-publish tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
