from __future__ import annotations

import importlib.util
import hashlib
import json
import tarfile
import io
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py"
)
LOAD_PUBLISH_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-load-publish-20260728-092500/"
    "model_runner_load_and_publish_preflight.json"
)
PUBLISHED_BINDING_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-published-binding-20260728-100419/"
    "model_runner_published_candidate_binding_preflight.json"
)
TP4_REPLAY_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-synthetic-binding-20260728-122021/"
    "tp4_synthetic_binding_oracle_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_real_candidate_provenance_replay_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisites_source_closure_and_frozen_methods():
    module = _load_module()
    prerequisites = module.load_tp4_real_candidate_prerequisites(
        LOAD_PUBLISH_ARTIFACT,
        PUBLISHED_BINDING_ARTIFACT,
        TP4_REPLAY_ARTIFACT,
    )

    assert prerequisites.load_publish_artifact_sha256 == (
        "d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18"
    )
    assert prerequisites.published_binding_artifact_sha256 == (
        "79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a"
    )
    assert prerequisites.tp4_replay_artifact_sha256 == (
        "803c8fac331eeee82b90013e0b0872de8f079661b6dd1ba43225fb446006cce4"
    )
    assert prerequisites.approved_model_manifest_sha256 == (
        "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
    )
    assert prerequisites.provenance == (
        "real-checkpoint-derived-serial-rank-replay"
    )
    assert prerequisites.claim_boundary == (
        "not-live-concurrent-tp4-candidate-binding"
    )
    assert module.PRODUCER_CONTEXTS == (
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    )
    assert module.MEMORY_CEILINGS_KIB == {
        "total": 6291456,
        "post_torch": 6029312,
        "post_metadata": 5767168,
    }
    assert len(prerequisites.inherited_source_file_sha256) == 57
    assert len(module.SOURCE_FILES) == 58
    assert len(set(module.SOURCE_FILES)) == 58
    assert set(module.SOURCE_FILES) - set(
        prerequisites.inherited_source_file_sha256
    ) == {
        "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py",
    }
    assert dict(module.AUTHORIZED_SOURCE_DELTAS) == {
        "tinyvllm/layers/linear.py": (
            "dba95145c0cc83b726694ddaae9de7a12206cacd3ecacc17403b3293cfe57b83",
            "9e4bbccd0fbaa4b901796884900a5ca203cbdeabce5049fdd655ddd7ad2bbcd8",
        ),
        "tinyvllm/models/qwen35_checkpoint_binding.py": (
            "9b54bdac2269ed943a2f7951ec03954c71c00b7f5aec8b9540fc4fde83d23012",
            "69578fe68404bfc6db58eac8664bd8cc23fcce84abe5f13cf9e9124fa2824b90",
        ),
        "tinyvllm/models/qwen35_components.py": (
            "c106f5598f5cb4f6af908089da233d5f20489195868c31c1fd1a532f9238ea3c",
            "93af914b4e957863b0df18ee99f6dba59120089bdb7ffe77fe32d5c11dcaa5c4",
        ),
    }
    assert module._validate_authorized_source_delta(
        module._source_hashes(ROOT),
        prerequisites.inherited_source_file_sha256,
    ) == {
        "tinyvllm/layers/linear.py": (
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/layers/linear.py"
            ][0],
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/layers/linear.py"
            ][1],
        ),
        "tinyvllm/models/qwen35_checkpoint_binding.py": (
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/models/qwen35_checkpoint_binding.py"
            ][0],
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/models/qwen35_checkpoint_binding.py"
            ][1],
        ),
        "tinyvllm/models/qwen35_components.py": (
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/models/qwen35_components.py"
            ][0],
            module.AUTHORIZED_SOURCE_DELTAS[
                "tinyvllm/models/qwen35_components.py"
            ][1],
        ),
    }

    methods = module.load_frozen_tp4_real_candidate_methods(ROOT)
    assert set(methods) == {
        "load_and_publish_qwen35_checkpoint_candidate",
        "bind_published_qwen35_loaded_checkpoint_candidate",
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
        "call_model_runner_acknowledged",
        "bind_qwen35_loaded_checkpoint_candidates",
    }
    assert all(callable(method) for method in methods.values())


def test_static_safety_and_serial_contract():
    module = _load_module()
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 58
    audit = module.audit_tp4_real_candidate_source(ROOT)
    assert audit == {
        "llm_engine_import_count": 0,
        "model_runner_import_count": 0,
        "llm_engine_construction_count": 0,
        "model_runner_construction_count": 0,
        "fixed_tinyvllm_shared_memory_count": 0,
        "scheduler_call_count": 0,
        "step_call_count": 0,
        "cuda_operation_call_count": 0,
        "forward_call_count": 0,
        "inference_call_count": 0,
        "authorized_loader_builder_call_count": 1,
        "producer_process_start_call_count": 1,
        "producer_process_join_call_count": 1,
    }


def _producer_row(module, rank=0):
    binding_hashes = [f"{index:064x}" for index in range(320)]
    phase_hashes = {
        f"phase_{index:02d}": f"{index + 500:064x}"
        for index in range(26)
    }
    return {
        "schema_version": module.PRODUCER_ROW_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": module.PROVENANCE,
        "claim_boundary": module.CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "process_id": 31_000_000 + rank,
        "observed_user": "sitian",
        "observed_hostname": "producer-test-host",
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "model_manifest_sha256": (
            module.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": (
            module.APPROVED_COMPOSITE_SHA256
        ),
        "authorization_sha256": module.AUTHORIZATION_SHA256,
        "model_runner_file_sha256": (
            module.real_binding_gate.MODEL_RUNNER_FILE_SHA256
        ),
        "method_source_sha256": dict(module.METHOD_SOURCE_SHA256),
        "metadata_bytes_read": 144024,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "load_publish_method_call_count": 1,
        "bind_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "production_publish_call_count": 1,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "binding_hash_count": 320,
        "binding_destination_sha256": binding_hashes,
        "phase_hash_count": 26,
        "phase_destination_sha256": phase_hashes,
        "aggregate_destination_sha256": "a" * 64,
        "aggregate_hash_verified": True,
        "alias_groups": (
            module.real_binding_gate.load_publish_gate.publication_gate
            .publication.ownership.loader_core.ALIAS_GROUPS
        ),
        "loader_stats": {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": module.MAX_TENSOR_BYTES,
        },
        "layout_fingerprint": "b" * 64,
        "dtype": "bfloat16",
        "method_row": {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": (
                module.APPROVED_MODEL_MANIFEST_SHA256
            ),
            "layout_fingerprint": "b" * 64,
            "dtype": "bfloat16",
            "detail": "",
        },
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_private_objects_collected": True,
        "collected_private_objects": {
            "runner": True,
            "production_slot": True,
            "request": True,
            "candidate": True,
            "owner": True,
            "runtime_bridge": True,
            "runtime_identity": True,
            "model": True,
            "pool": True,
            "target": True,
        },
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "memory": {
            "before": {"vmhwm_kib": 100, "vmrss_kib": 100},
            "after_torch": {"vmhwm_kib": 200, "vmrss_kib": 200},
            "after_metadata": {"vmhwm_kib": 300, "vmrss_kib": 300},
            "after_pool": {"vmhwm_kib": 400, "vmrss_kib": 400},
            "after_target": {"vmhwm_kib": 500, "vmrss_kib": 500},
            "after_clear": {"vmhwm_kib": 600, "vmrss_kib": 600},
        },
        "total_vmhwm_increment_kib": 500,
        "post_torch_vmhwm_increment_kib": 400,
        "post_metadata_vmhwm_increment_kib": 300,
    }


def test_tp4_real_producer_row_schema_and_rank_identity():
    module = _load_module()
    for rank in range(4):
        row = _producer_row(module, rank)
        assert module.validate_tp4_real_candidate_producer_row(row) is row

    invalid = _producer_row(module, 2)
    invalid["method_row"]["participant_id"] = 1
    try:
        module.validate_tp4_real_candidate_producer_row(invalid)
    except ValueError as error:
        assert "method row" in str(error)
    else:
        raise AssertionError("producer rank/participant mismatch must fail")


def test_candidate_payload_recorder_hashes_binding_phase_and_aggregate():
    module = _load_module()

    class Tensor:
        def __init__(self, payload):
            self.payload = payload

    bindings = [
        types.SimpleNamespace(
            destination=Tensor(index.to_bytes(4, "little")),
            destination_slice=None,
        )
        for index in range(320)
    ]
    target = types.SimpleNamespace(
        assembly=types.SimpleNamespace(
            packed=types.SimpleNamespace(model=object())
        ),
        binding_plan=types.SimpleNamespace(bindings=bindings),
        pool=object(),
    )
    candidate = types.SimpleNamespace(
        owner=types.SimpleNamespace(
            model=target.assembly.packed.model,
            pool=target.pool,
        ),
        binding_plan=target.binding_plan,
        model_fingerprint=module.APPROVED_MODEL_MANIFEST_SHA256,
    )
    result = module.record_tp4_loaded_candidate_payload(
        candidate=candidate,
        target=target,
        model_fingerprint=module.APPROVED_MODEL_MANIFEST_SHA256,
        tensor_bytes=lambda tensor: tensor.payload,
        destination_view=lambda binding: binding.destination,
    )

    expected_binding = [
        hashlib.sha256(index.to_bytes(4, "little")).hexdigest()
        for index in range(320)
    ]
    aggregate = hashlib.sha256()
    for index in range(320):
        aggregate.update(index.to_bytes(4, "little"))
    assert result["binding_destination_sha256"] == expected_binding
    assert result["binding_hash_count"] == 320
    assert result["phase_hash_count"] == 26
    assert len(result["phase_destination_sha256"]) == 26
    assert result["aggregate_destination_sha256"] == aggregate.hexdigest()
    assert result["aggregate_hash_verified"] is True


def test_tp4_real_candidate_producer_scope_loads_binds_records_then_clears():
    module = _load_module()

    calls = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Storage:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

    class Tensor:
        def __init__(self, value=0):
            self.value = value
            self.shape = (1,)
            self.dtype = "bfloat16"
            self.device = "cpu"
            self._storage = Storage(id(self))

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.value)

        def zero_(self):
            self.value = 0
            return self

        def fill_(self, value):
            self.value = value
            return self

        def count_nonzero(self):
            return Scalar(int(self.value != 0))

        def equal(self, other):
            return self.value == other.value

        def untyped_storage(self):
            return self._storage

        def storage_offset(self):
            return 0

    class Model:
        def __init__(self):
            self.values = [Tensor() for _ in range(296)]

        def named_parameters(self, remove_duplicate=False):
            return [
                (f"value_{index}", value)
                for index, value in enumerate(self.values)
            ]

        def named_buffers(self, remove_duplicate=False):
            return []

    class Request:
        authorization_sha256 = module.AUTHORIZATION_SHA256

    class Layout:
        fingerprint = "b" * 64

    class Pool:
        def __init__(self):
            self.layout = Layout()

    class Owner:
        def __init__(self, model, pool):
            self.model = model
            self.pool = pool
            self.runtime_bridge = RuntimeBridge()

    class Identity:
        model_fingerprint = module.APPROVED_MODEL_MANIFEST_SHA256
        layout_fingerprint = "b" * 64
        dtype = "bfloat16"

    class RuntimeBridge:
        pass

    class Candidate:
        def __init__(self, owner, binding_plan):
            self.owner = owner
            self.binding_plan = binding_plan
            self.model_fingerprint = (
                module.APPROVED_MODEL_MANIFEST_SHA256
            )
            self.stats = types.SimpleNamespace(
                assigned_bindings=320,
                source_tensors=320,
                shard_count=1,
                loaded_bytes=3763655360,
                peak_source_bytes=module.MAX_TENSOR_BYTES,
            )

    class Slot:
        def __init__(self):
            self.candidate = None
            self.publish_calls = 0

        def publish(self, candidate):
            self.publish_calls += 1
            self.candidate = candidate

    class Target:
        pass

    def private_graph_factory():
        calls.append("factory")
        model = Model()
        bindings = [
            types.SimpleNamespace(
                destination=model.values[index % 296],
                destination_slice=None,
            )
            for index in range(320)
        ]
        target = Target()
        target.assembly = types.SimpleNamespace(
            packed=types.SimpleNamespace(model=model)
        )
        target.binding_plan = types.SimpleNamespace(bindings=bindings)
        target.pool = Pool()
        target._consumed = False
        request = Request()

        def installed_loader(observed_request):
            calls.append("loader")
            assert observed_request is request
            for index, binding in enumerate(bindings):
                binding.destination.fill_(index + 1)
            target._consumed = True
            return Candidate(
                Owner(model, target.pool),
                target.binding_plan,
            )

        return target, request, installed_loader

    def load_and_publish(runner, observed_request):
        calls.append("load_and_publish")
        candidate = runner.qwen35_checkpoint_candidate_loader(
            observed_request
        )
        runner.qwen35_loaded_checkpoint_candidate_slot.publish(
            candidate
        )
        runner.qwen35_checkpoint_candidate_load_request = (
            observed_request
        )
        return {
            "participant_id": runner.rank,
            "operation": "load_checkpoint_candidate",
            "status": "published",
            "model_fingerprint": candidate.model_fingerprint,
            "detail": "",
        }

    def bind_owner(runner, owner):
        calls.append("bind_owner")
        runner.qwen35_hybrid_model_owner = owner
        runner.hybrid_state_runtime_bridge = owner.runtime_bridge

    def bind_candidate(runner, candidate):
        calls.append("bind_candidate")
        identity = Identity()
        runner.bind_qwen35_hybrid_model_owner(candidate.owner)
        runner.qwen35_hybrid_prefix_runtime_identity = identity
        runner.qwen35_hybrid_prefix_runtime_identity_owner = (
            candidate.owner
        )
        return {
            "participant_id": runner.rank,
            "model_fingerprint": identity.model_fingerprint,
            "layout_fingerprint": identity.layout_fingerprint,
            "dtype": "bfloat16",
        }

    def bind_published(runner):
        calls.append("bind_published")
        identity = runner.bind_qwen35_loaded_checkpoint_candidate(
            runner.qwen35_loaded_checkpoint_candidate_slot.candidate
        )
        return {
            **identity,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "detail": "",
        }

    def candidate_validator(**kwargs):
        calls.append("validate")
        target = kwargs["target"]
        assert (
            kwargs["candidate"].owner.model
            is target.assembly.packed.model
        )
        assert (
            kwargs["candidate"].binding_plan
            is target.binding_plan
        )
        return {"loaded_state_verified": True}

    def payload_recorder(**kwargs):
        calls.append("record")
        bindings = kwargs["target"].binding_plan.bindings
        assert all(
            int(binding.destination.count_nonzero().item()) == 1
            for binding in bindings
        )
        return module.record_tp4_loaded_candidate_payload(
            **kwargs,
            tensor_bytes=lambda tensor: tensor.value.to_bytes(
                4, "little"
            ),
            destination_view=lambda binding: binding.destination,
        )

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = nullcontext
    fake_torch.bfloat16 = "bfloat16"
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        result = module.execute_tp4_real_candidate_producer_scope(
            private_graph_factory=private_graph_factory,
            model_fingerprint=module.APPROVED_MODEL_MANIFEST_SHA256,
            methods={
                "load_and_publish_qwen35_checkpoint_candidate": (
                    load_and_publish
                ),
                "bind_published_qwen35_loaded_checkpoint_candidate": (
                    bind_published
                ),
            },
            bind_owner_method=bind_owner,
            bind_candidate_method=bind_candidate,
            production_slot_factory=Slot,
            candidate_validator=candidate_validator,
            payload_recorder=payload_recorder,
            rank=0,
        )
    finally:
        if previous_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous_torch

    assert calls == [
        "factory",
        "load_and_publish",
        "loader",
        "validate",
        "record",
        "bind_published",
        "bind_candidate",
        "bind_owner",
    ]
    assert result["load_publish_method_call_count"] == 1
    assert result["bind_method_call_count"] == 1
    assert result["owner_binding_method_call_count"] == 1
    assert result["candidate_binding_method_call_count"] == 1
    assert result["production_publish_call_count"] == 1
    assert result["binding_hash_count"] == 320
    assert result["phase_hash_count"] == 26
    assert result["method_row"]["status"] == "bound"
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["all_private_objects_collected"] is True
    assert all(result["collected_private_objects"].values())


def test_serial_tp4_real_candidate_coordinator_never_overlaps_workers():
    module = _load_module()
    events = []
    alive = set()
    maximum_alive = 0

    class Process:
        def __init__(self, rank):
            self.rank = rank
            self.pid = None
            self.exitcode = None

        def start(self):
            nonlocal maximum_alive
            events.append(("start", self.rank, tuple(sorted(alive))))
            assert not alive
            self.pid = 32_000_000 + self.rank
            alive.add(self.pid)
            maximum_alive = max(maximum_alive, len(alive))

        def join(self):
            events.append(("join", self.rank, tuple(sorted(alive))))
            assert alive == {self.pid}
            alive.remove(self.pid)
            self.exitcode = 0

    processes = {}

    def process_factory(tp_size, tp_rank):
        assert tp_size == 4
        process = Process(tp_rank)
        processes[tp_rank] = process
        return process

    def row_reader(tp_size, tp_rank, process):
        events.append(("read", tp_rank, tuple(sorted(alive))))
        assert tp_size == 4
        assert process is processes[tp_rank]
        assert process.pid not in alive
        return _producer_row(module, tp_rank) | {
            "process_id": process.pid,
        }

    rows = module.run_serial_tp4_real_candidate_producers(
        process_factory=process_factory,
        row_reader=row_reader,
        pid_is_alive=lambda pid: pid in alive,
    )

    assert tuple(row["tp_rank"] for row in rows) == (0, 1, 2, 3)
    assert tuple(row["process_id"] for row in rows) == (
        32_000_000,
        32_000_001,
        32_000_002,
        32_000_003,
    )
    assert maximum_alive == 1
    assert not alive
    assert events == [
        ("start", 0, ()),
        ("join", 0, (32_000_000,)),
        ("read", 0, ()),
        ("start", 1, ()),
        ("join", 1, (32_000_001,)),
        ("read", 1, ()),
        ("start", 2, ()),
        ("join", 2, (32_000_002,)),
        ("read", 2, ()),
        ("start", 3, ()),
        ("join", 3, (32_000_003,)),
        ("read", 3, ()),
    ]


def test_tp4_real_candidate_producer_worker_builds_strict_rank_row():
    module = _load_module()
    memory_points = iter([
        {"vmhwm_kib": 100, "vmrss_kib": 100},
        {"vmhwm_kib": 200, "vmrss_kib": 200},
        {"vmhwm_kib": 300, "vmrss_kib": 300},
        {"vmhwm_kib": 400, "vmrss_kib": 400},
        {"vmhwm_kib": 500, "vmrss_kib": 500},
        {"vmhwm_kib": 600, "vmrss_kib": 600},
    ])
    scope_result = _producer_row(module, 0)
    for name in (
        "schema_version",
        "status",
        "provenance",
        "claim_boundary",
        "tp_size",
        "tp_rank",
        "process_id",
        "observed_user",
        "observed_hostname",
        "checkpoint_dir",
        "model_manifest_sha256",
        "config_sha256",
        "index_sha256",
        "config_index_header_sha256",
        "authorization_sha256",
        "model_runner_file_sha256",
        "method_source_sha256",
        "metadata_bytes_read",
        "cuda_initialized_before",
        "cuda_initialized_after",
        "model_forward_count",
        "attention_forward_count",
        "memory",
        "total_vmhwm_increment_kib",
        "post_torch_vmhwm_increment_kib",
        "post_metadata_vmhwm_increment_kib",
    ):
        scope_result.pop(name)
    calls = []

    def runtime_factory(**kwargs):
        calls.append(("runtime", kwargs))
        return {
            "after_metadata": next(memory_points),
            "after_pool": next(memory_points),
            "after_target": next(memory_points),
            "metadata_bytes_read": 144024,
            "config_sha256": module.APPROVED_CONFIG_SHA256,
            "index_sha256": module.APPROVED_INDEX_SHA256,
            "config_index_header_sha256": (
                module.APPROVED_COMPOSITE_SHA256
            ),
            "model_forward_count": 0,
            "attention_forward_count": 0,
            "scope_kwargs": {"sentinel": "scope"},
        }

    def scope_executor(**kwargs):
        calls.append(("scope", kwargs))
        assert kwargs == {"sentinel": "scope"}
        return dict(scope_result)

    row = module.run_tp4_real_candidate_producer_worker(
        checkpoint_dir=module.APPROVED_MODEL_DIR,
        source_root=ROOT,
        tensor_parallel_size=4,
        tensor_parallel_rank=0,
        observed_user="sitian",
        observed_hostname="producer-test-host",
        process_id=31_000_000,
        status_reader=lambda: next(memory_points),
        torch_runtime=types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_initialized=lambda: False,
            ),
            set_num_threads=lambda count: calls.append(
                ("threads", count)
            ),
        ),
        runtime_factory=runtime_factory,
        scope_executor=scope_executor,
    )

    assert row == _producer_row(module, 0)
    assert [entry[0] for entry in calls] == [
        "threads",
        "runtime",
        "scope",
    ]
    assert calls[0] == ("threads", 8)
    assert calls[2] == ("scope", {"sentinel": "scope"})
    assert module.validate_tp4_real_candidate_producer_row(row) is row


def test_tp4_real_candidate_runtime_assembly_uses_tp4_rank_and_authorized_loader():
    module = _load_module()
    calls = []
    metadata = types.SimpleNamespace(
        metadata_bytes_read=144024,
        config_sha256=module.APPROVED_CONFIG_SHA256,
        index_sha256=module.APPROVED_INDEX_SHA256,
        config_index_header_sha256=module.APPROVED_COMPOSITE_SHA256,
        hf_config=object(),
        index_payload=object(),
        shard_headers=object(),
    )
    target = types.SimpleNamespace()
    request = object()
    adapter = object()

    components = {
        "read_metadata": lambda checkpoint_dir: (
            calls.append(("metadata", checkpoint_dir)) or metadata
        ),
        "build_tensor_plan": lambda observed_metadata: (
            calls.append(("plan", observed_metadata)) or "plan"
        ),
        "build_layout": lambda observed_metadata, tp_size: (
            calls.append(("layout", observed_metadata, tp_size))
            or "layout"
        ),
        "build_pool": lambda layout: (
            calls.append(("pool", layout)) or "pool"
        ),
        "prepare_target": (
            lambda observed_metadata, tensor_plan, pool, tp_size, tp_rank: (
                calls.append((
                    "target",
                    observed_metadata,
                    tensor_plan,
                    pool,
                    tp_size,
                    tp_rank,
                ))
                or target
            )
        ),
        "build_authorized_loader": lambda provider: (
            calls.append(("loader", provider())) or adapter
        ),
        "build_request": lambda checkpoint_dir: (
            calls.append(("request", checkpoint_dir)) or request
        ),
        "methods": {
            "load_and_publish_qwen35_checkpoint_candidate": object(),
            "bind_published_qwen35_loaded_checkpoint_candidate": object(),
        },
        "bind_owner_method": object(),
        "bind_candidate_method": object(),
        "production_slot_factory": object(),
        "candidate_validator": object(),
        "payload_recorder": object(),
    }
    points = iter([
        {"vmhwm_kib": 300, "vmrss_kib": 300},
        {"vmhwm_kib": 400, "vmrss_kib": 400},
        {"vmhwm_kib": 500, "vmrss_kib": 500},
    ])

    runtime = module.assemble_tp4_real_candidate_producer_runtime(
        checkpoint_dir=module.APPROVED_MODEL_DIR,
        tensor_parallel_size=4,
        tensor_parallel_rank=3,
        status_reader=lambda: next(points),
        components=components,
    )
    assert calls == []
    graph_target, graph_request, installed_loader = (
        runtime["scope_kwargs"]["private_graph_factory"]()
    )

    assert graph_target is target
    assert graph_request is request
    assert installed_loader is adapter
    assert runtime["after_metadata"]["vmhwm_kib"] == 300
    assert runtime["after_pool"]["vmhwm_kib"] == 400
    assert runtime["after_target"]["vmhwm_kib"] == 500
    assert runtime["scope_kwargs"]["methods"] is components["methods"]
    assert (
        runtime["scope_kwargs"]["bind_owner_method"]
        is components["bind_owner_method"]
    )
    assert runtime["scope_kwargs"]["rank"] == 3
    assert calls[-2:] == [
        ("loader", target),
        ("request", module.APPROVED_MODEL_DIR),
    ]
    assert ("layout", metadata, 4) in calls
    assert ("target", metadata, "plan", "pool", 4, 3) in calls


def test_production_component_factory_wires_frozen_methods_and_payload_helpers():
    module = _load_module()
    calls = []

    class Candidate:
        pass

    class Owner:
        pass

    class Slot:
        pass

    modules = {
        "tinyvllm.models.qwen35_checkpoint_metadata": types.SimpleNamespace(
            Qwen35CheckpointShardIdentity=lambda **kwargs: (
                "shard",
                kwargs,
            ),
            read_qwen35_checkpoint_metadata=lambda *args, **kwargs: (
                "metadata",
                args,
                kwargs,
            ),
        ),
        "tinyvllm.models.qwen35_checkpoint": types.SimpleNamespace(
            build_qwen35_checkpoint_tensor_plan=lambda *args: (
                "plan",
                args,
            ),
        ),
        "tinyvllm.engine.hybrid_state": types.SimpleNamespace(
            HybridStateTensorPool=lambda *args, **kwargs: (
                "pool",
                args,
                kwargs,
            ),
        ),
        "tinyvllm.engine.qwen35_hybrid_state": types.SimpleNamespace(
            build_qwen35_hybrid_state_layout=lambda *args, **kwargs: (
                "layout",
                args,
                kwargs,
            ),
        ),
        "tinyvllm.models.qwen35_checkpoint_candidate_factory": (
            types.SimpleNamespace(
                prepare_qwen35_checkpoint_candidate_target=(
                    lambda *args, **kwargs: ("target", args, kwargs)
                ),
            )
        ),
        "tinyvllm.models.qwen35_checkpoint_candidate_loader": (
            types.SimpleNamespace(
                build_qwen35_authorized_checkpoint_candidate_loader=(
                    lambda provider, **kwargs: (
                        "adapter",
                        provider,
                        kwargs,
                    )
                ),
            )
        ),
        "tinyvllm.models.qwen35_checkpoint_worker": types.SimpleNamespace(
            Qwen35CheckpointCandidateLoadRequest=lambda **kwargs: (
                "request",
                kwargs,
            ),
        ),
        "tinyvllm.models.qwen35_checkpoint_streaming": (
            types.SimpleNamespace(
                Qwen35LoadedCheckpointCandidate=Candidate,
            )
        ),
        "tinyvllm.engine.qwen35_hybrid_model_owner": (
            types.SimpleNamespace(Qwen35HybridModelOwner=Owner)
        ),
        "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity": (
            types.SimpleNamespace(
                bind_qwen35_hybrid_prefix_runtime_identity=object(),
            )
        ),
        "tinyvllm.engine.qwen35_hybrid_model_publication": (
            types.SimpleNamespace(
                Qwen35HybridModelOwnerPublicationSlot=Slot,
            )
        ),
    }

    def module_loader(name):
        calls.append(("module", name))
        return modules[name]

    frozen = {
        "load_and_publish_qwen35_checkpoint_candidate": object(),
        "bind_published_qwen35_loaded_checkpoint_candidate": object(),
    }
    binding = {
        "bind_qwen35_hybrid_model_owner": object(),
        "bind_qwen35_loaded_checkpoint_candidate": object(),
    }
    components = (
        module.build_tp4_real_candidate_producer_components(
            source_root=ROOT,
            module_loader=module_loader,
            torch_runtime=types.SimpleNamespace(
                bfloat16="bfloat16",
            ),
            backend_factory=lambda counter: ("backend", counter),
            frozen_method_loader=lambda root: (
                calls.append(("frozen", root)) or frozen
            ),
            binding_method_loader=lambda root, **kwargs: (
                calls.append(("binding", root, kwargs)) or binding
            ),
        )
    )

    assert set(components) == {
        "read_metadata",
        "build_tensor_plan",
        "build_layout",
        "build_pool",
        "prepare_target",
        "build_authorized_loader",
        "build_request",
        "methods",
        "bind_owner_method",
        "bind_candidate_method",
        "production_slot_factory",
        "candidate_validator",
        "payload_recorder",
    }
    assert components["methods"] is frozen
    assert (
        components["bind_owner_method"]
        is binding["bind_qwen35_hybrid_model_owner"]
    )
    assert (
        components["bind_candidate_method"]
        is binding["bind_qwen35_loaded_checkpoint_candidate"]
    )
    assert components["production_slot_factory"] is Slot
    assert calls[-2][0] == "frozen"
    assert calls[-1][0] == "binding"


def test_producer_worker_default_path_builds_source_bound_production_runtime():
    module = _load_module()
    calls = []
    memory_points = iter([
        {"vmhwm_kib": 100, "vmrss_kib": 100},
        {"vmhwm_kib": 200, "vmrss_kib": 200},
        {"vmhwm_kib": 600, "vmrss_kib": 600},
    ])
    scope_result = _producer_row(module, 0)
    for name in (
        "schema_version",
        "status",
        "provenance",
        "claim_boundary",
        "tp_size",
        "tp_rank",
        "process_id",
        "observed_user",
        "observed_hostname",
        "checkpoint_dir",
        "model_manifest_sha256",
        "config_sha256",
        "index_sha256",
        "config_index_header_sha256",
        "authorization_sha256",
        "model_runner_file_sha256",
        "method_source_sha256",
        "metadata_bytes_read",
        "cuda_initialized_before",
        "cuda_initialized_after",
        "model_forward_count",
        "attention_forward_count",
        "memory",
        "total_vmhwm_increment_kib",
        "post_torch_vmhwm_increment_kib",
        "post_metadata_vmhwm_increment_kib",
    ):
        scope_result.pop(name)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_initialized=lambda: False),
        set_num_threads=lambda count: calls.append(("threads", count)),
    )
    original_install = module._install_namespace_packages
    original_load = module._load_runtime_module
    original_components = (
        module.build_tp4_real_candidate_producer_components
    )
    original_assemble = (
        module.assemble_tp4_real_candidate_producer_runtime
    )
    module._install_namespace_packages = lambda root: calls.append(
        ("install", root)
    )
    module._load_runtime_module = lambda name: (
        calls.append(("import", name)) or fake_torch
    )
    module.build_tp4_real_candidate_producer_components = (
        lambda **kwargs: (
            calls.append(("components", kwargs)) or {"sentinel": True}
        )
    )
    def assemble_runtime(
        *,
        checkpoint_dir,
        tensor_parallel_size,
        tensor_parallel_rank,
        status_reader,
        components,
    ):
        kwargs = {
            "checkpoint_dir": checkpoint_dir,
            "tensor_parallel_size": tensor_parallel_size,
            "tensor_parallel_rank": tensor_parallel_rank,
            "status_reader": status_reader,
            "components": components,
        }
        calls.append(("assemble", kwargs))
        return {
                "after_metadata": {
                    "vmhwm_kib": 300,
                    "vmrss_kib": 300,
                },
                "after_pool": {
                    "vmhwm_kib": 400,
                    "vmrss_kib": 400,
                },
                "after_target": {
                    "vmhwm_kib": 500,
                    "vmrss_kib": 500,
                },
                "metadata_bytes_read": 144024,
                "config_sha256": module.APPROVED_CONFIG_SHA256,
                "index_sha256": module.APPROVED_INDEX_SHA256,
                "config_index_header_sha256": (
                    module.APPROVED_COMPOSITE_SHA256
                ),
                "model_forward_count": 0,
                "attention_forward_count": 0,
                "scope_kwargs": {"sentinel": "scope"},
        }

    module.assemble_tp4_real_candidate_producer_runtime = (
        assemble_runtime
    )
    try:
        row = module.run_tp4_real_candidate_producer_worker(
            checkpoint_dir=module.APPROVED_MODEL_DIR,
            source_root=ROOT,
            tensor_parallel_size=4,
            tensor_parallel_rank=0,
            observed_user="sitian",
            observed_hostname="producer-test-host",
            process_id=31_000_000,
            status_reader=lambda: next(memory_points),
            torch_runtime=None,
            runtime_factory=None,
            scope_executor=lambda **kwargs: (
                calls.append(("scope", kwargs))
                or dict(scope_result)
            ),
        )
    finally:
        module._install_namespace_packages = original_install
        module._load_runtime_module = original_load
        module.build_tp4_real_candidate_producer_components = (
            original_components
        )
        module.assemble_tp4_real_candidate_producer_runtime = (
            original_assemble
        )

    assert module.validate_tp4_real_candidate_producer_row(row) is row
    assert calls[0] == ("install", ROOT)
    assert calls[1] == ("import", "torch")
    assert calls[2] == ("threads", 8)
    assert calls[3][0] == "components"
    assert calls[4][0] == "assemble"
    assert calls[5] == ("scope", {"sentinel": "scope"})


def test_internal_producer_worker_atomically_writes_one_valid_rank_row():
    module = _load_module()
    row = _producer_row(module, 2)
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "rank2.json"

        result = module.write_tp4_real_candidate_producer_row(
            output_path=output,
            worker=lambda **kwargs: (
                calls.append(kwargs) or dict(row)
            ),
            checkpoint_dir=module.APPROVED_MODEL_DIR,
            source_root=ROOT,
            tensor_parallel_size=4,
            tensor_parallel_rank=2,
            observed_user="sitian",
            observed_hostname="producer-test-host",
            process_id=row["process_id"],
        )

        assert result == row
        assert json.loads(output.read_text()) == row
        assert sorted(Path(directory).iterdir()) == [output]
        assert calls == [{
            "checkpoint_dir": module.APPROVED_MODEL_DIR,
            "source_root": ROOT,
            "tensor_parallel_size": 4,
            "tensor_parallel_rank": 2,
            "observed_user": "sitian",
            "observed_hostname": "producer-test-host",
            "process_id": row["process_id"],
        }]


def test_rank_row_reader_rejects_pid_or_rank_drift():
    module = _load_module()
    row = _producer_row(module, 1)
    process = types.SimpleNamespace(pid=row["process_id"])
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "rank1.json"
        output.write_text(json.dumps(row, sort_keys=True))
        assert module.read_tp4_real_candidate_producer_row(
            output_path=output,
            tensor_parallel_size=4,
            tensor_parallel_rank=1,
            process=process,
        ) == row

        tampered = dict(row)
        tampered["process_id"] += 1
        output.write_text(json.dumps(tampered, sort_keys=True))
        try:
            module.read_tp4_real_candidate_producer_row(
                output_path=output,
                tensor_parallel_size=4,
                tensor_parallel_rank=1,
                process=process,
            )
        except ValueError as error:
            assert "identity" in str(error)
        else:
            raise AssertionError("producer PID drift must fail")


def test_real_candidate_provenance_oracle_finalizes_only_exited_serial_rows():
    module = _load_module()
    rows = [
        _producer_row(module, rank)
        for rank in range(4)
    ]
    for rank, row in enumerate(rows):
        row["process_id"] = 33_000_000 + rank
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "oracle.json"
        oracle = module.finalize_tp4_real_candidate_provenance_oracle(
            rows=rows,
            output_path=output,
            source_root=ROOT,
            pid_is_alive=lambda _pid: False,
        )

        assert oracle["schema_version"] == (
            module.PROVENANCE_ORACLE_SCHEMA_VERSION
        )
        assert oracle["status"] == "PASS"
        assert oracle["provenance"] == module.PROVENANCE
        assert oracle["claim_boundary"] == module.CLAIM_BOUNDARY
        assert oracle["producer_contexts"] == [
            [4, 0], [4, 1], [4, 2], [4, 3],
        ]
        assert oracle["producer_process_ids"] == [
            33_000_000,
            33_000_001,
            33_000_002,
            33_000_003,
        ]
        assert oracle["all_producers_exited_before_finalization"] is True
        assert oracle["producer_rows"] == rows
        assert oracle["producer_rows_sha256"] == hashlib.sha256(
            module._canonical(rows)
        ).hexdigest()
        assert json.loads(output.read_text()) == oracle
        assert module.validate_tp4_real_candidate_provenance_oracle(
            oracle
        ) is oracle

        try:
            module.finalize_tp4_real_candidate_provenance_oracle(
                rows=rows,
                output_path=output,
                source_root=ROOT,
                pid_is_alive=lambda _pid: False,
            )
        except ValueError as error:
            assert "exists" in str(error)
        else:
            raise AssertionError("immutable oracle overwrite must fail")


def test_real_candidate_provenance_oracle_rejects_live_or_partial_producers():
    module = _load_module()
    rows = [_producer_row(module, rank) for rank in range(4)]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "oracle.json"
        try:
            module.finalize_tp4_real_candidate_provenance_oracle(
                rows=rows[:3],
                output_path=output,
                source_root=ROOT,
                pid_is_alive=lambda _pid: False,
            )
        except ValueError as error:
            assert "complete" in str(error)
        else:
            raise AssertionError("partial producer set must fail")

        live_pid = rows[2]["process_id"]
        try:
            module.finalize_tp4_real_candidate_provenance_oracle(
                rows=rows,
                output_path=output,
                source_root=ROOT,
                pid_is_alive=lambda pid: pid == live_pid,
            )
        except RuntimeError as error:
            assert "alive" in str(error)
        else:
            raise AssertionError("live producer must fail")


def test_real_candidate_replay_cases_project_rows_and_limit_rank2_mutation():
    module = _load_module()
    rows = [_producer_row(module, rank) for rank in range(4)]
    oracle = {
        "schema_version": module.PROVENANCE_ORACLE_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": module.PROVENANCE,
        "claim_boundary": module.CLAIM_BOUNDARY,
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "model_manifest_sha256": module.APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": (
            module.APPROVED_COMPOSITE_SHA256
        ),
        "authorization_sha256": module.AUTHORIZATION_SHA256,
        "method_source_sha256": dict(module.METHOD_SOURCE_SHA256),
        "source_file_sha256": module._source_hashes(ROOT),
        "source_tree_sha256": module.tp4_gate._source_tree_sha256(
            module._source_hashes(ROOT)
        ),
        "producer_contexts": [[4, rank] for rank in range(4)],
        "producer_process_ids": [
            row["process_id"] for row in rows
        ],
        "all_producers_exited_before_finalization": True,
        "producer_rows_sha256": hashlib.sha256(
            module._canonical(rows)
        ).hexdigest(),
        "producer_rows": rows,
    }
    before = module._canonical(oracle["producer_rows"])
    cases = module.build_tp4_real_candidate_replay_cases(oracle)

    assert tuple(cases) == module.REPLAY_MODES
    baseline = cases[module.REPLAY_MODES[0]]
    assert baseline == tuple(row["method_row"] for row in rows)
    expected_fields = {
        module.REPLAY_MODES[1]: "model_fingerprint",
        module.REPLAY_MODES[2]: "layout_fingerprint",
        module.REPLAY_MODES[3]: "dtype",
    }
    for mode, field in expected_fields.items():
        case = cases[mode]
        differences = [
            (rank, name)
            for rank in range(4)
            for name in (
                "model_fingerprint",
                "layout_fingerprint",
                "dtype",
            )
            if case[rank][name] != baseline[rank][name]
        ]
        assert differences == [(2, field)]
        assert case[2][field] == (
            "float32"
            if field == "dtype"
            else hashlib.sha256(
                f"{field}-mismatch".encode()
            ).hexdigest()
        )
        assert all(
            case[rank][name] == baseline[rank][name]
            for rank in range(4)
            for name in (
                "participant_id",
                "operation",
                "status",
                "detail",
            )
        )
    assert module._canonical(oracle["producer_rows"]) == before


def test_real_candidate_success_replay_uses_production_tp4_transport_and_binder():
    module = _load_module()
    rows = [_producer_row(module, rank) for rank in range(4)]
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory) / "oracle.json"
        oracle = module.finalize_tp4_real_candidate_provenance_oracle(
            rows=rows,
            output_path=oracle_path,
            source_root=ROOT,
            pid_is_alive=lambda _pid: False,
        )
        attempt = module.execute_tp4_real_candidate_replay_attempt(
            source_root=ROOT,
            oracle=oracle,
            mode=module.REPLAY_MODES[0],
            timeout_s=5.0,
            name_prefix="q35-real-replay-test",
        )

    assert attempt["status"] == "PASS"
    assert attempt["mode"] == module.REPLAY_MODES[0]
    assert attempt["oracle_rows"] == [
        row["method_row"] for row in rows
    ]
    assert attempt["binding_rows"] == attempt["oracle_rows"]
    assert attempt["completion_configuration"] == [
        module.APPROVED_MODEL_MANIFEST_SHA256,
        rows[0]["layout_fingerprint"],
        "bfloat16",
        5.0,
    ]
    assert attempt["completion_committed"] is True
    assert attempt["repeat_zero_binding_dispatch"] is True
    assert attempt["ack_send_order"] == [3, 2, 1]
    assert attempt["collector_return_order"] == [1, 2, 3]
    assert attempt["ack_status_by_rank"] == {
        "1": "ok", "2": "ok", "3": "ok",
    }
    assert attempt["segment_unlinked"] is True
    assert attempt["post_unlink_attach_failed"] is True
    assert all(attempt["child_collected_by_rank"].values())


def test_real_candidate_directed_replays_change_one_rank2_field_only():
    module = _load_module()
    rows = [_producer_row(module, rank) for rank in range(4)]
    with tempfile.TemporaryDirectory() as directory:
        oracle = module.finalize_tp4_real_candidate_provenance_oracle(
            rows=rows,
            output_path=Path(directory) / "oracle.json",
            source_root=ROOT,
            pid_is_alive=lambda _pid: False,
        )
        producer_hash = oracle["producer_rows_sha256"]
        baseline = tuple(row["method_row"] for row in rows)
        expected = {
            module.REPLAY_MODES[1]: "model_fingerprint",
            module.REPLAY_MODES[2]: "layout_fingerprint",
            module.REPLAY_MODES[3]: "dtype",
        }
        for mode, field in expected.items():
            attempt = module.execute_tp4_real_candidate_replay_attempt(
                source_root=ROOT,
                oracle=oracle,
                mode=mode,
                timeout_s=5.0,
                name_prefix=f"q35-real-{field}-test",
            )
            differences = [
                (rank, name)
                for rank in range(4)
                for name in (
                    "model_fingerprint",
                    "layout_fingerprint",
                    "dtype",
                )
                if attempt["oracle_rows"][rank][name]
                != baseline[rank][name]
            ]
            assert differences == [(2, field)]
            assert attempt["authorized_changed_field"] == field
            assert attempt["producer_rows_sha256"] == producer_hash
            assert attempt["binding_rows"] is None
            assert attempt["completion_configuration"] is None
            assert attempt["completion_committed"] is False
            assert attempt["repeat_zero_binding_dispatch"] is False
            assert f"mismatch: {field}" in attempt["error_detail"]
            assert attempt["ack_status_by_rank"] == {
                "1": "ok", "2": "ok", "3": "ok",
            }
            assert attempt["collector_poisoned"] is False
            assert attempt["segment_unlinked"] is True
            assert attempt["post_unlink_attach_failed"] is True
            assert all(
                attempt["child_collected_by_rank"].values()
            )
        assert oracle["producer_rows_sha256"] == producer_hash
        assert module._canonical(oracle["producer_rows"]) == (
            module._canonical(rows)
        )


def test_replay_worker_row_and_final_result_preserve_oracle_identity():
    module = _load_module()
    rows = [_producer_row(module, rank) for rank in range(4)]
    with tempfile.TemporaryDirectory() as directory:
        oracle = module.finalize_tp4_real_candidate_provenance_oracle(
            rows=rows,
            output_path=Path(directory) / "oracle.json",
            source_root=ROOT,
            pid_is_alive=lambda _pid: False,
        )
        attempt_rows = []
        for index, mode in enumerate(module.REPLAY_MODES):
            attempt = {
                "status": "PASS",
                "mode": mode,
                "process_id": 34_000_000 + index,
                "child_process_ids": {
                    "1": 35_000_000 + index * 3,
                    "2": 35_000_001 + index * 3,
                    "3": 35_000_002 + index * 3,
                },
                "child_exitcodes": {"1": 0, "2": 0, "3": 0},
                "child_collected_by_rank": {
                    "1": True, "2": True, "3": True,
                },
                "shared_memory_name": f"real_replay_{index}",
                "shared_memory_capacity": 2**20,
                "segment_unlinked": True,
                "post_unlink_attach_failed": True,
                "dispatch_count": 2,
                "binding_dispatch_count": 1,
                "write_count": 2,
                "read_count_by_rank": {
                    "1": 2, "2": 2, "3": 2,
                },
                "executor_count_by_rank": {
                    "1": 2, "2": 2, "3": 2,
                },
                "event_set_count_by_rank": {
                    "1": 2, "2": 2, "3": 2,
                },
                "event_wait_count_by_rank": {
                    "1": 2, "2": 2, "3": 2,
                },
                "event_clear_count_by_rank": {
                    "1": 2, "2": 2, "3": 2,
                },
                "write_payload_bytes": [199, 154],
                "envelopes": [
                    {
                        "command_id": 0,
                        "method_name": (
                            "bind_published_qwen35_"
                            "loaded_checkpoint_candidate"
                        ),
                        "args": [],
                        "requires_ack": True,
                    },
                    {
                        "command_id": 1,
                        "method_name": "exit",
                        "args": [],
                        "requires_ack": False,
                    },
                ],
                "ack_send_order": [3, 2, 1],
                "collector_return_order": [1, 2, 3],
                "ack_status_by_rank": {
                    "1": "ok", "2": "ok", "3": "ok",
                },
                "collector_poisoned": False,
                "oracle_rows": [
                    dict(row)
                    for row in module.build_tp4_real_candidate_replay_cases(
                        oracle
                    )[mode]
                ],
                "authorized_changed_field": (
                    None
                    if index == 0
                    else (
                        "model_fingerprint",
                        "layout_fingerprint",
                        "dtype",
                    )[index - 1]
                ),
                "binding_rows": (
                    [dict(row["method_row"]) for row in rows]
                    if index == 0
                    else None
                ),
                "completion_configuration": (
                    [
                        module.APPROVED_MODEL_MANIFEST_SHA256,
                        rows[0]["layout_fingerprint"],
                        "bfloat16",
                        5.0,
                    ]
                    if index == 0
                    else None
                ),
                "completion_committed": index == 0,
                "repeat_zero_binding_dispatch": index == 0,
                "error_detail": (
                    ""
                    if index == 0
                    else "RuntimeError: loaded checkpoint candidate "
                    "binding mismatch: "
                    + (
                        "model_fingerprint",
                        "layout_fingerprint",
                        "dtype",
                    )[index - 1]
                ),
                "provenance": module.PROVENANCE,
                "claim_boundary": module.CLAIM_BOUNDARY,
                "producer_rows_sha256": (
                    oracle["producer_rows_sha256"]
                ),
                "provenance_oracle_sha256": hashlib.sha256(
                    module._canonical(oracle)
                ).hexdigest(),
            }
            row = module.build_tp4_real_candidate_replay_row(
                attempt=attempt,
                observed_user="sitian",
                observed_hostname="replay-test-host",
                process_id=attempt["process_id"],
            )
            assert module.validate_tp4_real_candidate_replay_row(
                row,
                oracle,
            ) is row
            attempt_rows.append(row)

        result = module.finalize_tp4_real_candidate_replay_result(
            oracle=oracle,
            replay_rows=attempt_rows,
            source_root=ROOT,
        )
        assert result["status"] == "PASS"
        assert result["replay_rows"] == attempt_rows
        assert result["producer_process_ids"] == (
            oracle["producer_process_ids"]
        )
        assert result["replay_outer_process_ids"] == [
            34_000_000,
            34_000_001,
            34_000_002,
            34_000_003,
        ]
        assert len(result["replay_child_process_ids"]) == 12
        assert result["all_replay_processes_distinct_from_producers"] is True


def test_source_tar_is_deterministic_and_cli_exposes_all_internal_modes():
    module = _load_module()
    first = module.build_source_tar(ROOT)
    second = module.build_source_tar(ROOT)
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == list(
            module.SOURCE_FILES
        )
        assert all(member.uid == 0 for member in members)
        assert all(member.gid == 0 for member in members)
        assert all(member.mtime == 0 for member in members)

    parser = module._parser()
    commands = parser._subparsers._group_actions[0].choices
    assert set(commands) == {
        "run",
        "internal-producer-worker",
        "internal-finalize-oracle",
        "internal-replay-worker",
        "internal-finalize-result",
        "validate",
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
        "qwen35 TP4 real-candidate provenance replay tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
