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
    "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py"
)
PREREQUISITE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-load-publish-20260728-092500/"
    "model_runner_load_and_publish_preflight.json"
)
COMPLETE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_model_runner_published_binding_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisite_source_closure_and_method_identities():
    module = _load_module()
    prerequisite = (
        module.load_model_runner_published_binding_prerequisite(
            PREREQUISITE_ARTIFACT
        )
    )

    assert prerequisite.artifact_sha256 == (
        "d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18"
    )
    assert prerequisite.source_tree_sha256 == (
        "a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8"
    )
    assert tuple(prerequisite.rows) == ((1, 0), (2, 0), (2, 1))
    assert len(prerequisite.source_file_sha256) == 50
    assert len(module.SOURCE_FILES) == 51
    assert len(set(module.SOURCE_FILES)) == 51
    assert set(module.SOURCE_FILES) - set(
        prerequisite.source_file_sha256
    ) == {
        "tools/"
        "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py"
    }

    methods = module.load_frozen_model_runner_published_binding_methods(
        ROOT,
        owner_type=object,
        candidate_type=object,
        identity_binder=lambda owner, fingerprint: None,
    )
    assert set(methods) == {
        "publish_qwen35_loaded_checkpoint_candidate",
        "bind_qwen35_hybrid_model_owner",
        "bind_qwen35_loaded_checkpoint_candidate",
        "bind_published_qwen35_loaded_checkpoint_candidate",
    }
    assert all(callable(method) for method in methods.values())


def test_flat_prerequisite_row_reconstructs_canonical_candidate_oracle():
    module = _load_module()
    prerequisite = (
        module.load_model_runner_published_binding_prerequisite(
            PREREQUISITE_ARTIFACT
        )
    )
    complete = json.loads(COMPLETE_ARTIFACT.read_text())
    for tp_size, tp_rank in module.PREREQUISITE_ROWS:
        flat = (
            module.select_model_runner_published_binding_prerequisite_row(
                prerequisite,
                tp_size,
                tp_rank,
            )
        )
        oracle = module.reconstruct_candidate_validation_oracle(flat)
        canonical = next(
            row
            for row in complete["rows"]
            if row["tp_size"] == tp_size
            and row["tp_rank"] == tp_rank
        )
        assert oracle["binding_results"] == [
            {
                "binding_index": row["binding_index"],
                "phase_name": row["phase_name"],
                "destination_sha256": row[
                    "destination_sha256"
                ],
            }
            for row in canonical["binding_results"]
        ]
        assert oracle["phase_results"] == [
            {
                "phase_name": row["phase_name"],
                "destination_sha256": row[
                    "destination_sha256"
                ],
            }
            for row in canonical["phase_results"]
        ]
        assert oracle["aggregate_destination_sha256"] == (
            canonical["aggregate_destination_sha256"]
        )


def test_extracted_methods_compose_success_and_conflict():
    module = _load_module()

    class Owner:
        pass

    class Candidate:
        pass

    class Identity:
        def rank_row(self, participant_id):
            return {
                "participant_id": participant_id,
                "model_fingerprint": "a" * 64,
                "layout_fingerprint": "layout-a",
                "dtype": "bfloat16",
            }

    def bind_identity(owner, fingerprint):
        assert type(owner) is Owner
        assert fingerprint == "a" * 64
        return Identity()

    methods = module.load_frozen_model_runner_published_binding_methods(
        ROOT,
        owner_type=Owner,
        candidate_type=Candidate,
        identity_binder=bind_identity,
    )

    class Slot:
        candidate = None

        def publish(self, candidate):
            self.candidate = candidate

    def runner(existing_bridge=None):
        model = SimpleNamespace()
        pool = SimpleNamespace()
        runtime_bridge = SimpleNamespace(pool=pool)
        state_transaction = SimpleNamespace(pool=pool)
        layer_stack = SimpleNamespace(state_transaction=state_transaction)
        model.layer_stack = layer_stack
        owner = Owner()
        owner.model = model
        owner.pool = pool
        owner.runtime_bridge = runtime_bridge
        owner.state_transaction = state_transaction
        owner.layer_stack = layer_stack
        candidate = Candidate()
        candidate.owner = owner
        candidate.model_fingerprint = "a" * 64
        shell = SimpleNamespace(
            rank=1,
            model=model,
            qwen35_loaded_checkpoint_candidate_slot=Slot(),
            hybrid_state_runtime_bridge=existing_bridge,
            qwen35_hybrid_model_owner=None,
            qwen35_hybrid_prefix_restore_owner=None,
            qwen35_hybrid_prefix_restore_participant=None,
            qwen35_hybrid_prefix_publication_participant=None,
            qwen35_hybrid_prefix_runtime_identity=None,
            qwen35_hybrid_prefix_runtime_identity_owner=None,
        )
        shell.bind_qwen35_hybrid_model_owner = (
            lambda value: methods["bind_qwen35_hybrid_model_owner"](
                shell, value
            )
        )
        shell.bind_qwen35_loaded_checkpoint_candidate = (
            lambda value: methods["bind_qwen35_loaded_checkpoint_candidate"](
                shell, value
            )
        )
        return shell, candidate

    shell, candidate = runner()
    assert methods["publish_qwen35_loaded_checkpoint_candidate"](
        shell, candidate
    ) is candidate
    row = methods[
        "bind_published_qwen35_loaded_checkpoint_candidate"
    ](shell)
    assert row == {
        "participant_id": 1,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound",
        "model_fingerprint": "a" * 64,
        "layout_fingerprint": "layout-a",
        "dtype": "bfloat16",
        "detail": "",
    }
    assert shell.qwen35_hybrid_model_owner is candidate.owner
    assert shell.hybrid_state_runtime_bridge is (
        candidate.owner.runtime_bridge
    )
    assert shell.qwen35_hybrid_prefix_runtime_identity_owner is (
        candidate.owner
    )

    conflict = SimpleNamespace()
    shell, candidate = runner(conflict)
    methods["publish_qwen35_loaded_checkpoint_candidate"](
        shell, candidate
    )
    row = methods[
        "bind_published_qwen35_loaded_checkpoint_candidate"
    ](shell)
    assert row == {
        "participant_id": 1,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "error",
        "model_fingerprint": "",
        "layout_fingerprint": "",
        "dtype": "",
        "detail": (
            "RuntimeError: a different hybrid state runtime bridge "
            "is already installed"
        ),
    }
    assert shell.hybrid_state_runtime_bridge is conflict
    assert shell.qwen35_hybrid_model_owner is None
    assert shell.qwen35_hybrid_prefix_runtime_identity is None
    assert shell.qwen35_hybrid_prefix_runtime_identity_owner is None


def _binding_row(module, mode):
    prerequisite = json.loads(PREREQUISITE_ARTIFACT.read_text())
    inherited = next(
        row
        for row in prerequisite["rows"]
        if row["tp_size"] == 1
        and row["tp_rank"] == 0
        and row["mode"] == "success"
    )
    row = dict(inherited)
    memory = dict(row["memory"])
    memory["after_binding_clear"] = memory.pop(
        "after_load_publish_clear"
    )
    row.update({
        "schema_version": module.ROW_SCHEMA_VERSION,
        "mode": mode,
        "prerequisite_artifact_sha256": (
            module.PREREQUISITE_ARTIFACT_SHA256
        ),
        "model_runner_file_sha256": module.MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(
            module.METHOD_SOURCE_SHA256
        ),
        "publication_method_call_count": 1,
        "outer_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "production_publish_call_count": 1,
        "production_slot_visibility_verified": True,
        "published_candidate_identity_verified": True,
        "candidate_installed": mode == "success",
        "owner_binding_visible": mode == "success",
        "runtime_bridge_binding_visible": mode == "success",
        "runtime_identity_binding_visible": mode == "success",
        "runtime_identity_owner_visible": mode == "success",
        "injected_bridge_preserved": (
            mode == "injected_bridge_conflict"
        ),
        "binding_state_pristine": (
            mode == "injected_bridge_conflict"
        ),
        "layout_fingerprint": "layout-a",
        "dtype": "bfloat16",
        "method_row": {
            "participant_id": 0,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": (
                "bound"
                if mode == "success"
                else "error"
            ),
            "model_fingerprint": (
                module.APPROVED_MODEL_MANIFEST_SHA256
                if mode == "success"
                else ""
            ),
            "layout_fingerprint": (
                "layout-a" if mode == "success" else ""
            ),
            "dtype": "bfloat16" if mode == "success" else "",
            "detail": (
                ""
                if mode == "success"
                else (
                    "RuntimeError: a different hybrid state runtime "
                    "bridge is already installed"
                )
            ),
        },
        "collected_private_objects": {
            "runner": True,
            "production_slot": True,
            "candidate": True,
            "owner": True,
            "runtime_bridge": True,
            "model": True,
            "pool": True,
            "target": True,
            **({
                "runtime_identity": True,
            } if mode == "success" else {
                "injected_bridge": True,
            }),
        },
        "all_private_binding_objects_collected": True,
        "memory": memory,
    })
    return row


def test_rank_row_contract_accepts_success_and_conflict():
    module = _load_module()
    for mode in ("success", "injected_bridge_conflict"):
        row = _binding_row(module, mode)
        assert (
            module.validate_model_runner_published_binding_row(row)
            is row
        )

    damaged = _binding_row(module, "injected_bridge_conflict")
    damaged["owner_binding_visible"] = True
    try:
        module.validate_model_runner_published_binding_row(damaged)
    except ValueError as error:
        assert "owner_binding_visible" in str(error)
    else:
        raise AssertionError("damaged conflict row was accepted")


def test_rank_row_contract_rejects_damaged_real_checkpoint_evidence():
    module = _load_module()
    damage_cases = (
        ("observed_user", "nobody"),
        ("checkpoint_dir", "/tmp/not-approved"),
        ("config_sha256", "0" * 64),
        ("index_sha256", "0" * 64),
        ("config_index_header_sha256", "0" * 64),
        ("metadata_bytes_read", 0),
        ("selected_binding_count", 319),
        ("unique_destination_count", 295),
        ("alias_groups", ()),
        ("target_consumed_before", True),
        ("target_consumed_after", False),
        ("loader_stats", {}),
        ("binding_destination_sha256", ["short"] * 320),
        ("phase_destination_sha256", {}),
        ("aggregate_destination_sha256", "short"),
    )
    for name, value in damage_cases:
        damaged = _binding_row(module, "success")
        damaged[name] = value
        try:
            module.validate_model_runner_published_binding_row(damaged)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError(
                f"damaged real-checkpoint field was accepted: {name}"
            )


def test_private_published_binding_transaction_success_and_conflict():
    module = _load_module()
    selected = torch.full((2,), 9, dtype=torch.bfloat16)
    expected = torch.tensor((1.0, 2.0), dtype=torch.bfloat16)

    class Value:
        pass

    class Owner:
        pass

    class Candidate:
        pass

    class Identity:
        def __init__(self, owner):
            self.owner = owner
            self.model_fingerprint = "a" * 64
            self.layout_fingerprint = "layout-a"
            self.dtype = torch.bfloat16

        def rank_row(self, participant_id):
            return {
                "participant_id": participant_id,
                "model_fingerprint": self.model_fingerprint,
                "layout_fingerprint": self.layout_fingerprint,
                "dtype": "bfloat16",
            }

        def __eq__(self, other):
            return (
                type(other) is Identity
                and other.owner is self.owner
                and other.model_fingerprint == self.model_fingerprint
                and other.layout_fingerprint == self.layout_fingerprint
                and other.dtype == self.dtype
            )

    def bind_identity(owner, fingerprint):
        assert type(owner) is Owner
        assert fingerprint == "a" * 64
        return Identity(owner)

    methods = module.load_frozen_model_runner_published_binding_methods(
        ROOT,
        owner_type=Owner,
        candidate_type=Candidate,
        identity_binder=bind_identity,
    )

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
            self.publish_calls = 0

        def publish(self, candidate):
            self.publish_calls += 1
            self.candidate = candidate
            self.owner = candidate.owner
            self.model_fingerprint = candidate.model_fingerprint

    value_hash = module._sha256(
        expected.view(torch.uint8).numpy().tobytes()
    )
    oracle = {
        "binding_results": [{
            "binding_index": 0,
            "phase_name": "all",
            "destination_sha256": value_hash,
        }],
        "phase_results": [{
            "phase_name": "all",
            "destination_sha256": value_hash,
        }],
        "aggregate_destination_sha256": value_hash,
    }

    def private_graph_factory():
        model = Model()
        pool = Value()
        pool.marker = "pool"
        pool.layout = SimpleNamespace(
            fingerprint="layout-a",
            components=(
                SimpleNamespace(dtype=torch.bfloat16),
            ),
        )
        state_transaction = Value()
        state_transaction.pool = pool
        layer_stack = Value()
        layer_stack.state_transaction = state_transaction
        model.layer_stack = layer_stack
        runtime_bridge = Value()
        runtime_bridge.pool = pool
        owner = Owner()
        owner.model = model
        owner.pool = pool
        owner.layer_stack = layer_stack
        owner.state_transaction = state_transaction
        owner.runtime_bridge = runtime_bridge
        plan = Value()
        plan.bindings = (Binding(),)
        target = Value()
        target.assembly = Value()
        target.assembly.packed = Value()
        target.assembly.packed.model = model
        target.binding_plan = plan
        target.pool = pool
        target._consumed = False
        candidate = Candidate()
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

        def loader():
            target._consumed = True
            selected.copy_(expected)
            return candidate

        return target, loader

    def validate_candidate(*, candidate, target, model_fingerprint, oracle_row):
        assert candidate.owner.model is target.assembly.packed.model
        assert model_fingerprint == "a" * 64
        assert oracle_row is oracle
        assert torch.equal(selected, expected)
        return {
            "loaded_state_verified": True,
            "binding_destination_sha256": [value_hash],
            "binding_hash_count": 1,
            "phase_destination_sha256": {"all": value_hash},
            "phase_hash_count": 1,
            "aggregate_destination_sha256": value_hash,
            "aggregate_hash_verified": True,
        }

    for mode in ("success", "injected_bridge_conflict"):
        result = module.execute_model_runner_published_binding_scope(
            private_graph_factory=private_graph_factory,
            oracle_row=oracle,
            model_fingerprint="a" * 64,
            methods=methods,
            production_slot_factory=Slot,
            candidate_validator=validate_candidate,
            mode=mode,
            rank=2,
        )
        assert result["publication_method_call_count"] == 1
        assert result["outer_binding_method_call_count"] == 1
        assert result["candidate_binding_method_call_count"] == 1
        assert result["owner_binding_method_call_count"] == 1
        assert result["adapter_call_count"] == 1
        assert result["provider_call_count"] == 1
        assert result["production_publish_call_count"] == 1
        assert result["production_slot_visibility_verified"] is True
        assert result["all_private_binding_objects_collected"] is True
        assert not int(selected.count_nonzero().item())
        if mode == "success":
            assert result["method_row"]["status"] == "bound"
            assert result["owner_binding_visible"] is True
            assert result["runtime_bridge_binding_visible"] is True
            assert result["runtime_identity_binding_visible"] is True
            assert result["runtime_identity_owner_visible"] is True
        else:
            assert result["method_row"]["status"] == "error"
            assert result["method_row"]["detail"] == (
                "RuntimeError: a different hybrid state runtime "
                "bridge is already installed"
            )
            assert result["injected_bridge_preserved"] is True
            assert result["binding_state_pristine"] is True


def test_orchestration_contract_source_closure_and_partial_rejection():
    module = _load_module()
    assert module.SCHEMA_VERSION == (
        "qwen35.real-checkpoint-model-runner-published-binding.v1"
    )
    assert module.WORKER_CONTEXTS == (
        (1, 0, "success"),
        (1, 0, "injected_bridge_conflict"),
        (2, 0, "success"),
        (2, 0, "injected_bridge_conflict"),
        (2, 1, "success"),
        (2, 1, "injected_bridge_conflict"),
    )
    prerequisite = (
        module.load_model_runner_published_binding_prerequisite(
            PREREQUISITE_ARTIFACT
        )
    )
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 51
    assert {
        name: hashes[name]
        for name in prerequisite.source_file_sha256
    } == dict(prerequisite.source_file_sha256)
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


def test_static_safety_audit_rejects_forbidden_execution_paths():
    module = _load_module()
    audit = module.audit_published_binding_preflight_source(ROOT)
    assert audit == {
        "adapter_builder_call_count": 1,
        "production_slot_constructor_call_count": 1,
        "extracted_method_invocation_count": {
            name: 1 for name in module.METHOD_SOURCE_SHA256
        },
        "model_runner_import_count": 0,
        "model_runner_construction_count": 0,
        "direct_streamed_loader_call_count": 0,
        "target_take_call_count": 0,
        "engine_call_count": 0,
        "scheduler_call_count": 0,
        "forward_call_count": 0,
        "cuda_execution_call_count": 0,
        "cuda_is_initialized_call_count": 2,
    }


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 ModelRunner published candidate binding tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
