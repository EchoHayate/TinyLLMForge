from __future__ import annotations

import hashlib
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
    "qwen35_real_checkpoint_private_publication_slot_preflight.py"
)
COMPLETE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)
OWNERSHIP_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-private-ownership-20260728-090000/"
    "private_candidate_ownership_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_private_publication_slot_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def test_exact_prerequisite_oracles_and_source_closure():
    module = _load_module()
    prerequisites = module.load_private_publication_prerequisites(
        COMPLETE_ARTIFACT,
        OWNERSHIP_ARTIFACT,
    )
    assert prerequisites.complete_artifact_sha256 == (
        "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
    )
    assert prerequisites.ownership_artifact_sha256 == (
        "977a20a1986ade81e2b94063287cd15e6ece2adc3c818f3e0d9589f75b1adac4"
    )
    assert prerequisites.ownership_source_tree_sha256 == (
        "91f9225a6ee214049002dc12bc7a669cdfa6a0d847b03e0cc107834f96f561a0"
    )
    assert tuple(prerequisites.complete_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert tuple(prerequisites.ownership_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert len(prerequisites.ownership_source_file_sha256) == 45
    assert len(module.SOURCE_FILES) == 47
    assert len(set(module.SOURCE_FILES)) == 47
    assert set(module.SOURCE_FILES) - set(
        prerequisites.ownership_source_file_sha256
    ) == {
        "tinyvllm/engine/qwen35_hybrid_model_publication.py",
        "tools/qwen35_real_checkpoint_private_publication_slot_preflight.py",
    }


def _synthetic_transaction(expected_error_message=None):
    module = _load_module()
    selected = torch.full((4,), 9, dtype=torch.bfloat16)
    rotary = torch.tensor((0.5, 0.25), dtype=torch.float32)
    expected = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16
    )
    oracle = {
        "binding_results": [
            {
                "binding_index": 0,
                "phase_name": "left",
                "destination_sha256": _digest(expected[:2]),
            },
            {
                "binding_index": 1,
                "phase_name": "right",
                "destination_sha256": _digest(expected[2:]),
            },
        ],
        "phase_results": [
            {
                "phase_name": "left",
                "destination_sha256": _digest(expected[:2]),
            },
            {
                "phase_name": "right",
                "destination_sha256": _digest(expected[2:]),
            },
        ],
        "aggregate_destination_sha256": _digest(expected),
    }

    class Value:
        pass

    class Binding:
        pass

    class Model:
        def named_parameters(self, remove_duplicate=False):
            return (("selected", selected),)

        def named_buffers(self, remove_duplicate=False):
            return (("rotary", rotary),)

    class Slot:
        def __init__(self):
            self.candidate = None
            self.owner = None
            self.model_fingerprint = None
            self.publish_calls = 0

        def publish(self, value):
            self.publish_calls += 1
            self.candidate = value
            self.owner = value.owner
            self.model_fingerprint = value.model_fingerprint
            return value.owner

    def private_graph_factory():
        bindings = []
        for offset in (0, 2):
            binding = Binding()
            binding.destination = selected
            binding.destination_slice = (offset, 2)
            bindings.append(binding)
        binding_plan = Value()
        binding_plan.bindings = tuple(bindings)
        model = Model()
        pool = Value()
        pool.marker = "pool"
        owner = Value()
        owner.model = model
        owner.layer_stack = Value()
        owner.state_transaction = Value()
        owner.pool = pool
        owner.runtime_bridge = Value()
        owner.runtime_bridge.pool = pool
        target = Value()
        target.assembly = Value()
        target.assembly.packed = Value()
        target.assembly.packed.model = model
        target.binding_plan = binding_plan
        target.pool = pool
        target._consumed = False
        candidate = Value()
        candidate.owner = owner
        candidate.binding_plan = binding_plan
        candidate.model_fingerprint = "a" * 64
        candidate.stats = SimpleNamespace(
            assigned_bindings=2,
            source_tensors=2,
            shard_count=1,
            loaded_bytes=8,
            peak_source_bytes=4,
        )

        def acquire_candidate():
            target._consumed = True
            selected.copy_(expected)
            return candidate

        return target, acquire_candidate

    result = module.execute_private_publication_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=oracle,
        model_fingerprint="a" * 64,
        publication_slot_factory=Slot,
        expected_error_message=expected_error_message,
    )
    return result, selected, rotary


def test_successful_private_publication_scope_is_collected():
    result, selected, rotary = _synthetic_transaction()
    assert result["publication_call_count"] == 1
    assert result["slot_empty_before_publication"] is True
    assert result["published_candidate_identity_verified"] is True
    assert result["published_owner_identity_verified"] is True
    assert result["published_fingerprint_verified"] is True
    assert result["binding_destination_sha256"] == [
        _digest(torch.tensor((1.0, 2.0), dtype=torch.bfloat16)),
        _digest(torch.tensor((3.0, 4.0), dtype=torch.bfloat16)),
    ]
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["all_private_publication_objects_collected"] is True
    assert result["collected_private_objects"] == {
        "slot": True,
        "candidate": True,
        "owner": True,
        "model": True,
        "pool": True,
        "target": True,
    }
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        rotary,
        torch.tensor((0.5, 0.25), dtype=torch.float32),
    )


def test_injected_post_publication_failure_is_collected():
    result, selected, rotary = _synthetic_transaction(
        "injected private publication-slot failure"
    )
    assert result["publication_call_count"] == 1
    assert result["published_candidate_identity_verified"] is True
    assert result["expected_failure_observed"] is True
    assert result["expected_failure_type"] == "RuntimeError"
    assert result["expected_failure_message"] == (
        "injected private publication-slot failure"
    )
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["all_private_publication_objects_collected"] is True
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        rotary,
        torch.tensor((0.5, 0.25), dtype=torch.float32),
    )


def test_real_worker_contract_constants():
    module = _load_module()
    assert module.SCHEMA_VERSION == (
        "qwen35.real-checkpoint-private-publication-slot.v1"
    )
    assert module.ROW_SCHEMA_VERSION == (
        "qwen35.real-checkpoint-private-publication-slot-rank.v1"
    )
    assert module.WORKER_CONTEXTS == (
        (1, 0, "success"),
        (1, 0, "injected_post_publication_failure"),
        (2, 0, "success"),
        (2, 0, "injected_post_publication_failure"),
        (2, 1, "success"),
        (2, 1, "injected_post_publication_failure"),
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
    assert module.PUBLICATION_MODULE_SHA256 == (
        "4ab2f928a3bbeeb632ca4180dcd496d56ac7716ac90d2a6adeb861f9c65d5b84"
    )


def test_source_closure_and_partial_finalization_rejection():
    module = _load_module()
    hashes = module._source_hashes(ROOT)
    ownership_record = json.loads(OWNERSHIP_ARTIFACT.read_text())
    assert len(hashes) == 47
    assert {
        name: hashes[name]
        for name in ownership_record["source_file_sha256"]
    } == ownership_record["source_file_sha256"]
    assert hashes[
        "tinyvllm/engine/qwen35_hybrid_model_publication.py"
    ] == module.PUBLICATION_MODULE_SHA256
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
        test_exact_prerequisite_oracles_and_source_closure,
        test_successful_private_publication_scope_is_collected,
        test_injected_post_publication_failure_is_collected,
        test_real_worker_contract_constants,
        test_source_closure_and_partial_finalization_rejection,
    )
    for test in tests:
        test()
    print(
        "qwen35 private publication-slot tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
