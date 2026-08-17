from __future__ import annotations

import ast
from dataclasses import asdict, replace
import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.config import Config
from tinyvllm.engine.autoregressive_draft_registration import (
    AutoregressiveDraftRegistrationCandidate,
    AutoregressiveDraftRegistrationError,
    CheckpointFingerprint,
    TokenizerContract,
    build_autoregressive_draft_registration_status,
    validate_autoregressive_draft_registration_consensus,
)
from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftTensorParallelCoordinator,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalExecutorRegistry,
    assert_tensor_free,
)
from tinyvllm.engine.speculative_runtime import (
    ModelRunnerProposalExecutorDescriptor,
)
from tinyvllm.speculative.adapter import DraftCapabilities


MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"
CONFIG_MODULE = importlib.import_module("tinyvllm.config")


def _config(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        CONFIG_MODULE.AutoConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(
            model_type="qwen3_5",
            max_position_embeddings=4096,
            num_hidden_layers=2,
        ),
    )
    target_path = tmp_path / "target"
    target_path.mkdir()
    values = {
        "model": str(target_path),
        "autoregressive_draft_enabled": False,
        "autoregressive_draft_model": None,
        "autoregressive_draft_backend": "qwen3",
        "autoregressive_draft_max_proposal_tokens": 4,
        "autoregressive_draft_gpu_slot_capacity": 0,
        "autoregressive_draft_proposal_kv_offload_enabled": False,
        "autoregressive_draft_logical_entry_capacity": 0,
        "autoregressive_draft_cpu_backing_capacity": 0,
    }
    values.update(overrides)
    return Config(**values)


def test_autoregressive_draft_config_defaults(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    assert config.autoregressive_draft_enabled is False
    assert config.autoregressive_draft_model is None
    assert config.autoregressive_draft_backend == "qwen3"
    assert config.autoregressive_draft_max_proposal_tokens == 4
    assert config.autoregressive_draft_gpu_slot_capacity == 0
    assert (
        config.autoregressive_draft_proposal_kv_offload_enabled
        is False
    )
    assert config.autoregressive_draft_logical_entry_capacity == 0
    assert config.autoregressive_draft_cpu_backing_capacity == 0


def test_autoregressive_draft_proposal_kv_offload_config_accepts_valid_shape(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        monkeypatch,
        autoregressive_draft_enabled=True,
        autoregressive_draft_model="draft",
        autoregressive_draft_gpu_slot_capacity=8,
        autoregressive_draft_proposal_kv_offload_enabled=True,
        autoregressive_draft_logical_entry_capacity=16,
        autoregressive_draft_cpu_backing_capacity=16,
    )

    assert config.autoregressive_draft_logical_entry_capacity == 16


@pytest.mark.parametrize("tensor_parallel_size", (1, 4))
def test_enabled_config_accepts_tp1_or_tp4(
    tmp_path,
    monkeypatch,
    tensor_parallel_size,
):
    config = _config(
        tmp_path,
        monkeypatch,
        autoregressive_draft_enabled=True,
        autoregressive_draft_model="draft",
        autoregressive_draft_gpu_slot_capacity=8,
        tensor_parallel_size=tensor_parallel_size,
    )

    assert config.tensor_parallel_size == tensor_parallel_size


@pytest.mark.parametrize(
    "tensor_parallel_size",
    (2, 3, 5, 8),
)
def test_enabled_config_rejects_unsupported_tp(
    tmp_path,
    monkeypatch,
    tensor_parallel_size,
):
    with pytest.raises(ValueError, match="1 or 4"):
        _config(
            tmp_path,
            monkeypatch,
            autoregressive_draft_enabled=True,
            autoregressive_draft_model="draft",
            autoregressive_draft_gpu_slot_capacity=8,
            tensor_parallel_size=tensor_parallel_size,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"autoregressive_draft_enabled": 1}, "enabled"),
        (
            {
                "autoregressive_draft_enabled": True,
                "autoregressive_draft_model": "",
                "autoregressive_draft_gpu_slot_capacity": 8,
            },
            "model",
        ),
        (
            {"autoregressive_draft_backend": "other"},
            "backend",
        ),
        (
            {"autoregressive_draft_max_proposal_tokens": 0},
            "max_proposal_tokens",
        ),
        (
            {"autoregressive_draft_max_proposal_tokens": 5},
            "max_proposal_tokens",
        ),
        (
            {
                "autoregressive_draft_enabled": True,
                "autoregressive_draft_model": "draft",
                "autoregressive_draft_gpu_slot_capacity": 0,
            },
            "slot_capacity",
        ),
        (
            {"autoregressive_draft_gpu_slot_capacity": -1},
            "slot_capacity",
        ),
        (
            {
                "autoregressive_draft_proposal_kv_offload_enabled": 1,
            },
            "offload_enabled",
        ),
        (
            {
                "autoregressive_draft_proposal_kv_offload_enabled": True,
                "autoregressive_draft_logical_entry_capacity": 16,
                "autoregressive_draft_gpu_slot_capacity": 8,
                "autoregressive_draft_cpu_backing_capacity": 16,
            },
            "requires autoregressive draft",
        ),
        (
            {
                "autoregressive_draft_enabled": True,
                "autoregressive_draft_model": "draft",
                "autoregressive_draft_proposal_kv_offload_enabled": True,
                "autoregressive_draft_logical_entry_capacity": 16,
                "autoregressive_draft_gpu_slot_capacity": 8,
                "autoregressive_draft_cpu_backing_capacity": 15,
            },
            "logical == cpu",
        ),
        (
            {
                "autoregressive_draft_enabled": True,
                "autoregressive_draft_model": "draft",
                "autoregressive_draft_proposal_kv_offload_enabled": True,
                "autoregressive_draft_logical_entry_capacity": 8,
                "autoregressive_draft_gpu_slot_capacity": 8,
                "autoregressive_draft_cpu_backing_capacity": 8,
            },
            "logical == cpu > gpu > 0",
        ),
    ),
)
def test_autoregressive_draft_config_rejects_invalid_values(
    tmp_path,
    monkeypatch,
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, monkeypatch, **overrides)


class _FakeExecutor:

    capabilities = DraftCapabilities(
        source_type="independent_draft_model",
        supports_batch=True,
        requires_target_hidden=False,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
        requires_proposal_lifecycle=True,
        requires_full_token_history=False,
    )

    def __init__(self, *, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def propose_batch(self, rows):
        return ()

    def observe_target_prefill(self, rows):
        return None

    def prepare_finalize_batch(self, rows):
        return "ticket"

    def commit_finalize_batch(self, ticket_id):
        return None

    def rollback_finalize_batch(self, ticket_id):
        return None

    def release_sequence(self, sequence_id, *, sequence_epoch):
        return None

    def authority_snapshot(self):
        return {
            "source_type": "independent_draft_model",
            "tensor_parallel_rank": self.rank,
            "tensor_parallel_size": self.world_size,
            "rank": self.rank,
            "world_size": self.world_size,
        }


class _ExistingMTPExecutor(_FakeExecutor):
    capabilities = DraftCapabilities(
        source_type="native_checkpoint_mtp",
        supports_batch=True,
        requires_target_hidden=True,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
        requires_proposal_lifecycle=True,
        requires_full_token_history=False,
    )


def test_registry_preflight_validates_without_mutation():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()

    normalized = registry.preflight_registration(
        "autoregressive-draft",
        executor,
        executor.capabilities,
    )

    assert normalized == executor.capabilities
    assert registry.lifecycle_executor_ids() == ()


def test_registry_preflight_and_register_reject_same_invalid_executor():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()
    executor.prepare_finalize_batch = None

    with pytest.raises(
        ValueError,
        match="prepare_finalize_batch",
    ):
        registry.preflight_registration(
            "autoregressive-draft",
            executor,
            executor.capabilities,
        )
    with pytest.raises(
        ValueError,
        match="prepare_finalize_batch",
    ):
        registry.register(
            "autoregressive-draft",
            executor,
            executor.capabilities,
        )
    assert registry.lifecycle_executor_ids() == ()


def test_registry_preflight_rejects_duplicate_id_without_mutation():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()
    registry.register(
        "autoregressive-draft",
        executor,
        executor.capabilities,
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.preflight_registration(
            "autoregressive-draft",
            _FakeExecutor(),
            _FakeExecutor.capabilities,
        )

    assert registry.lifecycle_executor_ids() == (
        "autoregressive-draft",
    )
    assert (
        registry.capabilities_for("autoregressive-draft")
        == executor.capabilities
    )


@pytest.mark.parametrize(
    ("executor_id", "executor", "capabilities", "message"),
    (
        ("", _FakeExecutor(), _FakeExecutor.capabilities, "ID"),
        (
            "autoregressive-draft",
            object(),
            _FakeExecutor.capabilities,
            "capabilities",
        ),
        (
            "autoregressive-draft",
            _FakeExecutor(),
            object(),
            "capabilities",
        ),
    ),
)
def test_registry_preflight_matches_register_validation(
    executor_id,
    executor,
    capabilities,
    message,
):
    preflight_registry = ModelRunnerProposalExecutorRegistry()
    register_registry = ModelRunnerProposalExecutorRegistry()

    with pytest.raises((TypeError, ValueError), match=message):
        preflight_registry.preflight_registration(
            executor_id,
            executor,
            capabilities,
        )
    with pytest.raises((TypeError, ValueError), match=message):
        register_registry.register(
            executor_id,
            executor,
            capabilities,
        )

    assert preflight_registry.lifecycle_executor_ids() == ()
    assert register_registry.lifecycle_executor_ids() == ()


class _Dependencies:

    def __init__(
        self,
        *,
        compatibility_error=None,
        model_error=None,
        rank=0,
        world_size=1,
        fail_stage=None,
    ):
        self.calls = []
        self.compatibility_error = compatibility_error
        self.model_error = model_error
        self.rank = rank
        self.world_size = world_size
        self.fail_stage = fail_stage
        self.model = SimpleNamespace(name="draft")
        self.store = SimpleNamespace(name="store")
        self.allocator = SimpleNamespace(
            name="allocator",
            physical_store=self.store,
        )
        self.cache = SimpleNamespace(name="cache")
        self.backend = SimpleNamespace(
            name="backend",
            backend_identity="qwen3",
        )
        self.graph_components = SimpleNamespace(
            scratch_cache=object(),
            scratch_owner=object(),
            backend=object(),
            runner=object(),
        )
        self.expect_graph_runner = False
        self.executor = _FakeExecutor(
            rank=rank,
            world_size=world_size,
        )

    def _fail(self, stage):
        if self.fail_stage == stage:
            raise RuntimeError(f"{stage} injected failure")

    def build_checkpoint_fingerprint(self, path):
        label = Path(path).name
        self.calls.append(f"fingerprint:{label}")
        self._fail(
            f"fingerprint_{label}_checkpoint"
        )
        return CheckpointFingerprint(
            model_path=str(Path(path).resolve()),
            config_sha256=f"{label}-config",
            shard_sha256=(("model.safetensors", f"{label}-shard"),),
            composite_sha256=f"{label}-composite",
        )

    def load_tokenizer(self, path):
        label = Path(path).name
        self.calls.append(f"load_tokenizer:{label}")
        self._fail(f"load_{label}_tokenizer")
        return SimpleNamespace(label=label)

    def build_tokenizer_contract(self, path, tokenizer):
        label = Path(path).name
        self.calls.append(f"tokenizer_contract:{label}")
        self._fail(
            f"build_{label}_tokenizer_contract"
        )
        return TokenizerContract(
            model_path=str(Path(path).resolve()),
            tokenizer_class="FixtureTokenizer",
            normalization_sha256="normalization",
            ordered_token_to_id_sha256="mapping",
            vocab_size=8,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            stop_token_ids=(1,),
            artifact_sha256=(),
            composite_sha256=f"{label}-tokenizer",
        )

    def validate_tokenizer_compatibility(self, target, draft):
        self.calls.append("validate_tokenizers")
        self._fail("validate_tokenizer_compatibility")
        if self.compatibility_error is not None:
            raise self.compatibility_error

    def load_hf_config(self, path):
        self.calls.append("load_hf_config")
        self._fail("load_draft_hf_config")
        return SimpleNamespace(model_type="qwen3")

    def build_model(
        self,
        hf_config,
        *,
        tensor_parallel_rank,
        tensor_parallel_size,
    ):
        assert tensor_parallel_rank == self.rank
        assert tensor_parallel_size == self.world_size
        self.calls.append("build_model")
        self._fail("build_draft_model")
        if self.model_error is not None:
            raise self.model_error
        return self.model

    def load_weights(self, model, path):
        assert model is self.model
        self.calls.append("load_weights")
        self._fail("load_draft_weights")

    def move_model_to_target(self, model, target):
        assert model is self.model
        self.calls.append("move_eval_model")
        self._fail("move_and_eval_draft_model")
        return torch.device("cpu"), torch.float32

    def build_proposal_kv_allocator(
        self,
        model,
        *,
        offload_enabled,
        logical_entry_capacity,
        gpu_slot_capacity,
        cpu_backing_capacity,
        async_copy,
        batch_copy,
        dtype,
        device,
    ):
        assert model is self.model
        assert offload_enabled is False
        assert logical_entry_capacity == 16
        assert gpu_slot_capacity == 16
        assert cpu_backing_capacity == 16
        assert async_copy is False
        assert batch_copy is True
        assert dtype == torch.float32
        assert device == torch.device("cpu")
        self.calls.append("build_allocator")
        self._fail("build_proposal_kv_allocator")
        return self.allocator

    def build_proposal_kv_cache(self, allocator):
        assert allocator is self.allocator
        self.calls.append("build_cache")
        self._fail("build_proposal_kv_cache")
        return self.cache

    def build_backend(
        self,
        *,
        model,
        proposal_kv_cache,
        backend_identity,
        model_fingerprint,
        tokenizer_fingerprint,
        tensor_parallel_rank,
        tensor_parallel_size,
    ):
        assert model is self.model
        assert proposal_kv_cache is self.cache
        assert backend_identity == "qwen3"
        assert model_fingerprint == "draft-composite"
        assert tokenizer_fingerprint == "draft-tokenizer"
        assert tensor_parallel_rank == self.rank
        assert tensor_parallel_size == self.world_size
        self.calls.append("build_backend")
        self._fail("build_qwen3_draft_backend")
        return self.backend

    def build_graph_components(
        self,
        *,
        config,
        backend,
        proposal_kv_cache,
        physical_store,
        device,
        dtype,
    ):
        assert config.autoregressive_draft_cuda_graphs is True
        assert backend is self.backend
        assert proposal_kv_cache is self.cache
        assert physical_store is self.store
        assert device == torch.device("cpu")
        assert dtype == torch.float32
        self.calls.append("build_graph_components")
        self._fail("build_autoregressive_draft_graph")
        return self.graph_components

    def build_executor(
        self,
        *,
        backend,
        proposal_kv_cache,
        max_proposal_tokens,
        tensor_parallel_rank,
        tensor_parallel_size,
        tensor_parallel_coordinator,
        graph_runner=None,
    ):
        assert backend is self.backend
        assert proposal_kv_cache is self.cache
        assert max_proposal_tokens == 4
        assert tensor_parallel_rank == self.rank
        assert tensor_parallel_size == self.world_size
        assert tensor_parallel_coordinator is not None
        assert graph_runner is (
            self.graph_components.runner
            if self.expect_graph_runner
            else None
        )
        self.calls.append("build_executor")
        self._fail("build_autoregressive_draft_executor")
        return self.executor

    def build_descriptor(self, executor):
        assert executor is self.executor
        self.calls.append("build_descriptor")
        self._fail("build_executor_descriptor")
        return ModelRunnerProposalExecutorDescriptor(
            executor_id="autoregressive-draft",
            capabilities=executor.capabilities,
        )


class _OffloadDependencies(_Dependencies):

    def __init__(self):
        super().__init__()
        self.store = SimpleNamespace(name="residency-storage")
        self.allocator = SimpleNamespace(
            name="residency-allocator",
            storage=self.store,
        )
        self.allocator_arguments = None

    def build_proposal_kv_allocator(
        self,
        model,
        *,
        offload_enabled,
        logical_entry_capacity,
        gpu_slot_capacity,
        cpu_backing_capacity,
        async_copy,
        batch_copy,
        dtype,
        device,
    ):
        assert model is self.model
        self.calls.append("build_allocator")
        self.allocator_arguments = {
            "offload_enabled": offload_enabled,
            "logical_entry_capacity": logical_entry_capacity,
            "gpu_slot_capacity": gpu_slot_capacity,
            "cpu_backing_capacity": cpu_backing_capacity,
            "async_copy": async_copy,
            "batch_copy": batch_copy,
            "dtype": dtype,
            "device": device,
        }
        return self.allocator


class _TrackingRegistry(ModelRunnerProposalExecutorRegistry):

    def __init__(self, calls, *, fail_stage=None):
        super().__init__()
        self.calls = calls
        self.fail_stage = fail_stage

    def preflight_registration(
        self,
        executor_id,
        executor,
        capabilities,
    ):
        self.calls.append("registry_preflight")
        if self.fail_stage == "registry_preflight":
            raise RuntimeError(
                "registry_preflight injected failure"
            )
        return super().preflight_registration(
            executor_id,
            executor,
            capabilities,
        )

    def register(self, executor_id, executor, capabilities):
        self.calls.append("register_executor")
        return super().register(
            executor_id,
            executor,
            capabilities,
        )


class _RegistrationCoordinator:

    def __init__(self, *, rank, calls, fail_stage=None):
        self.rank = rank
        self.world_size = 4
        self.calls = calls
        self.fail_stage = fail_stage

    @classmethod
    def matching(cls, rank, calls, *, fail_stage=None):
        return cls(
            rank=rank,
            calls=calls,
            fail_stage=fail_stage,
        )

    def collect_registration_status(self, local_status):
        self.calls.append("collect_registration_status")
        if self.fail_stage == "registration_consensus":
            raise RuntimeError(
                f"rank {self.rank} registration_consensus "
                "injected failure"
            )
        if not local_status.success:
            successful = replace(
                local_status,
                success=True,
                stage="ready",
                error_type=None,
                message=None,
                target_checkpoint_sha256="target",
                draft_checkpoint_sha256="draft",
                target_tokenizer_sha256="target-tokenizer",
                draft_tokenizer_sha256="draft-tokenizer",
                backend_identity="qwen3",
                executor_id="autoregressive-draft",
                capabilities_sha256="capabilities",
            )
            statuses = [
                replace(successful, rank=rank)
                for rank in range(self.world_size)
            ]
            statuses[self.rank] = local_status
            return tuple(statuses)
        return tuple(
            replace(local_status, rank=rank)
            for rank in range(self.world_size)
        )


def _model_runner_shell():
    module = ast.parse(MODEL_RUNNER_PATH.read_text())
    model_runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method_names = {
        "_maybe_register_autoregressive_draft_executor",
        "autoregressive_draft_authority_snapshot",
    }
    methods = [
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in method_names
    ]
    shell = ast.ClassDef(
        name="ModelRunnerRegistrationShell",
        bases=[],
        keywords=[],
        body=methods or [ast.Pass()],
        decorator_list=[],
    )
    namespace = {
        "asdict": asdict,
        "assert_tensor_free": assert_tensor_free,
        "torch": torch,
        "AutoregressiveDraftRegistrationCandidate": (
            AutoregressiveDraftRegistrationCandidate
        ),
        "AutoregressiveDraftRegistrationError": (
            AutoregressiveDraftRegistrationError
        ),
        "AutoregressiveDraftTensorParallelCoordinator": (
            AutoregressiveDraftTensorParallelCoordinator
        ),
        "build_autoregressive_draft_registration_status": (
            build_autoregressive_draft_registration_status
        ),
        "validate_autoregressive_draft_registration_consensus": (
            validate_autoregressive_draft_registration_consensus
        ),
        "_autoregressive_draft_registration_dependencies": (
            lambda: None
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[shell], type_ignores=[])
            ),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["ModelRunnerRegistrationShell"]


def _runner(
    tmp_path,
    *,
    enabled=True,
    proposal_kv_offload_enabled=False,
    tensor_parallel_size=1,
    rank=0,
    register_mtp=False,
    graph_enabled=False,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir(exist_ok=True)
    draft.mkdir(exist_ok=True)
    runner = _model_runner_shell()()
    runner.config = SimpleNamespace(
        model=str(target),
        autoregressive_draft_enabled=enabled,
        autoregressive_draft_model=str(draft),
        autoregressive_draft_backend="qwen3",
        autoregressive_draft_max_proposal_tokens=4,
        autoregressive_draft_gpu_slot_capacity=16,
        autoregressive_draft_proposal_kv_offload_enabled=(
            proposal_kv_offload_enabled
        ),
        autoregressive_draft_logical_entry_capacity=32,
        autoregressive_draft_cpu_backing_capacity=32,
        proposal_kv_async_copy=False,
        proposal_kv_batch_copy=True,
        tensor_parallel_size=tensor_parallel_size,
        autoregressive_draft_cuda_graphs=graph_enabled,
        autoregressive_draft_cuda_graph_q_allowlist=(4,),
        autoregressive_draft_cuda_graph_batch_allowlist=(4,),
        autoregressive_draft_cuda_graph_min_observations=2,
        autoregressive_draft_cuda_graph_max_entries=4,
        autoregressive_draft_cuda_graph_max_static_bytes=(
            64 * 1024 * 1024
        ),
        autoregressive_draft_cuda_graph_max_reserved_bytes=(
            512 * 1024 * 1024
        ),
        autoregressive_draft_cuda_graph_max_total_capture_ns=(
            5_000_000_000
        ),
        autoregressive_draft_cuda_graph_max_single_capture_ns=(
            2_000_000_000
        ),
    )
    runner.rank = rank
    runner.world_size = tensor_parallel_size
    runner.model = SimpleNamespace(name="target")
    runner.speculative_proposal_executors = (
        ModelRunnerProposalExecutorRegistry()
    )
    runner.autoregressive_draft_model = None
    runner.autoregressive_draft_physical_store = None
    runner.autoregressive_draft_executor = None
    runner.autoregressive_draft_executor_descriptor = None
    runner.autoregressive_draft_registration_error = None
    runner.autoregressive_draft_registration_consensus_sha256 = None
    runner.autoregressive_draft_checkpoint_identity = None
    runner.autoregressive_draft_tokenizer_contract = None
    runner.autoregressive_draft_graph_components = None
    if register_mtp:
        mtp = _ExistingMTPExecutor()
        runner.speculative_proposal_executors.register(
            "native_checkpoint_proposal",
            mtp,
            mtp.capabilities,
        )
    return runner


def test_disabled_config_performs_no_registration_work(tmp_path):
    runner = _runner(tmp_path, enabled=False)
    dependencies = _Dependencies()

    result = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert result is None
    assert dependencies.calls == []
    assert (
        runner.speculative_proposal_executors
        .lifecycle_executor_ids()
        == ()
    )


def test_default_off_builds_no_graph_components(tmp_path):
    runner = _runner(tmp_path)
    dependencies = _Dependencies()

    descriptor = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert "build_graph_components" not in dependencies.calls
    assert runner.autoregressive_draft_graph_components is None


def test_enabled_tp4_wires_graph_components_before_executor(tmp_path):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=4,
        graph_enabled=True,
    )
    dependencies = _Dependencies(rank=0, world_size=4)
    dependencies.expect_graph_runner = True
    coordinator = _RegistrationCoordinator.matching(
        0,
        dependencies.calls,
    )

    descriptor = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
        tensor_parallel_coordinator=coordinator,
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert dependencies.calls.index(
        "build_graph_components"
    ) < dependencies.calls.index("build_executor")
    assert runner.autoregressive_draft_graph_components is (
        dependencies.graph_components
    )


def test_graph_mode_rejects_offload_before_dependency_work(tmp_path):
    runner = _runner(
        tmp_path,
        proposal_kv_offload_enabled=True,
        tensor_parallel_size=4,
        graph_enabled=True,
    )
    dependencies = _Dependencies(rank=0, world_size=4)

    with pytest.raises(
        RuntimeError,
        match="graph.*offload",
    ):
        runner._maybe_register_autoregressive_draft_executor(
            registration_dependencies=dependencies,
        )

    assert dependencies.calls == []
    assert runner.autoregressive_draft_executor is None


@pytest.mark.parametrize(
    "tensor_parallel_size",
    (2, 3, 5, 8),
)
def test_enabled_unsupported_tp_fails_before_dependencies_are_called(
    tmp_path,
    tensor_parallel_size,
):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    dependencies = _Dependencies()

    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        runner._maybe_register_autoregressive_draft_executor(
            registration_dependencies=dependencies,
        )

    assert dependencies.calls == []
    assert (
        runner.speculative_proposal_executors
        .lifecycle_executor_ids()
        == ()
    )


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_privately_constructs_local_candidate_before_publication(
    tmp_path,
    rank,
):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=4,
        rank=rank,
    )
    dependencies = _Dependencies(rank=rank, world_size=4)
    runner.speculative_proposal_executors = _TrackingRegistry(
        dependencies.calls
    )
    coordinator = _RegistrationCoordinator.matching(
        rank,
        dependencies.calls,
    )

    descriptor = (
        runner._maybe_register_autoregressive_draft_executor(
            registration_dependencies=dependencies,
            tensor_parallel_coordinator=coordinator,
        )
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert dependencies.calls.index("registry_preflight") < (
        dependencies.calls.index("collect_registration_status")
    )
    assert dependencies.calls.index(
        "collect_registration_status"
    ) < dependencies.calls.index("register_executor")


@pytest.mark.parametrize(
    "failing_stage",
    (
        "fingerprint_target_checkpoint",
        "fingerprint_draft_checkpoint",
        "load_target_tokenizer",
        "build_target_tokenizer_contract",
        "load_draft_tokenizer",
        "build_draft_tokenizer_contract",
        "validate_tokenizer_compatibility",
        "load_draft_hf_config",
        "build_draft_model",
        "load_draft_weights",
        "move_and_eval_draft_model",
        "build_proposal_kv_allocator",
        "build_proposal_kv_cache",
        "build_qwen3_draft_backend",
        "build_autoregressive_draft_executor",
        "build_executor_descriptor",
        "registry_preflight",
        "registration_consensus",
    ),
)
def test_tp4_registration_failure_publishes_no_partial_state(
    tmp_path,
    failing_stage,
):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=4,
        rank=0,
    )
    dependency_failure = (
        failing_stage
        if failing_stage
        not in ("registry_preflight", "registration_consensus")
        else None
    )
    dependencies = _Dependencies(
        rank=0,
        world_size=4,
        fail_stage=dependency_failure,
    )
    registry_failure = (
        failing_stage
        if failing_stage == "registry_preflight"
        else None
    )
    registry = _TrackingRegistry(
        dependencies.calls,
        fail_stage=registry_failure,
    )
    existing = _ExistingMTPExecutor()
    registry.register(
        "native_checkpoint_proposal",
        existing,
        existing.capabilities,
    )
    dependencies.calls.clear()
    runner.speculative_proposal_executors = registry
    coordinator = _RegistrationCoordinator.matching(
        0,
        dependencies.calls,
        fail_stage=(
            failing_stage
            if failing_stage == "registration_consensus"
            else None
        ),
    )

    result = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
        tensor_parallel_coordinator=coordinator,
    )

    assert result is None
    assert registry.lifecycle_executor_ids() == (
        "native_checkpoint_proposal",
    )
    assert runner.autoregressive_draft_model is None
    assert runner.autoregressive_draft_physical_store is None
    assert runner.autoregressive_draft_executor is None
    assert runner.autoregressive_draft_executor_descriptor is None
    assert runner.autoregressive_draft_checkpoint_identity is None
    assert runner.autoregressive_draft_tokenizer_contract is None
    assert (
        runner.autoregressive_draft_registration_consensus_sha256
        is None
    )
    error = runner.autoregressive_draft_registration_error
    assert isinstance(error, AutoregressiveDraftRegistrationError)
    assert "rank 0" in error.message
    assert failing_stage in error.message


def test_tp1_registration_uses_exact_dependency_order(tmp_path):
    runner = _runner(tmp_path, register_mtp=True)
    dependencies = _Dependencies()

    descriptor = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert dependencies.calls == [
        "fingerprint:target",
        "fingerprint:draft",
        "load_tokenizer:target",
        "tokenizer_contract:target",
        "load_tokenizer:draft",
        "tokenizer_contract:draft",
        "validate_tokenizers",
        "load_hf_config",
        "build_model",
        "load_weights",
        "move_eval_model",
        "build_allocator",
        "build_cache",
        "build_backend",
        "build_executor",
        "build_descriptor",
    ]
    assert (
        runner.speculative_proposal_executors
        .lifecycle_executor_ids()
        == (
            "native_checkpoint_proposal",
            "autoregressive-draft",
        )
    )
    assert runner.autoregressive_draft_model is dependencies.model
    assert (
        runner.autoregressive_draft_physical_store
        is dependencies.store
    )
    assert runner.autoregressive_draft_executor is dependencies.executor


def test_offload_registration_builds_residency_allocator_and_publishes_storage(
    tmp_path,
):
    runner = _runner(
        tmp_path,
        proposal_kv_offload_enabled=True,
    )
    dependencies = _OffloadDependencies()

    descriptor = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert dependencies.allocator_arguments == {
        "offload_enabled": True,
        "logical_entry_capacity": 32,
        "gpu_slot_capacity": 16,
        "cpu_backing_capacity": 32,
        "async_copy": False,
        "batch_copy": True,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    assert (
        runner.autoregressive_draft_physical_store
        is dependencies.store
    )
    assert runner.autoregressive_draft_executor is dependencies.executor


def test_failed_compatibility_allocates_no_model_or_slots(tmp_path):
    runner = _runner(tmp_path)
    dependencies = _Dependencies(
        compatibility_error=ValueError("mapping mismatch"),
    )

    result = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert result is None
    assert "build_model" not in dependencies.calls
    assert "build_allocator" not in dependencies.calls
    assert runner.autoregressive_draft_model is None
    assert runner.autoregressive_draft_physical_store is None
    assert isinstance(
        runner.autoregressive_draft_registration_error,
        AutoregressiveDraftRegistrationError,
    )


def test_failed_model_load_preserves_target_and_existing_mtp(tmp_path):
    runner = _runner(tmp_path, register_mtp=True)
    target_model = runner.model
    dependencies = _Dependencies(
        model_error=RuntimeError("load failed"),
    )

    result = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    assert result is None
    assert runner.model is target_model
    assert (
        runner.speculative_proposal_executors
        .lifecycle_executor_ids()
        == ("native_checkpoint_proposal",)
    )
    assert runner.autoregressive_draft_executor is None


def test_authority_snapshot_contains_identity_without_tensors(tmp_path):
    runner = _runner(tmp_path)
    dependencies = _Dependencies()
    runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )

    snapshot = runner.autoregressive_draft_authority_snapshot()

    assert_tensor_free(
        snapshot,
        name="autoregressive draft registration snapshot",
    )
    assert snapshot["registered"] is True
    assert snapshot["rank"] == 0
    assert snapshot["world_size"] == 1
    assert len(snapshot["registration_consensus_sha256"]) == 64
    assert (
        snapshot["checkpoint_identity"]["draft"][
            "composite_sha256"
        ]
        == "draft-composite"
    )
    assert (
        snapshot["tokenizer_contract"]["draft"][
            "composite_sha256"
        ]
        == "draft-tokenizer"
    )
    assert snapshot["executor"]["source_type"] == (
        "independent_draft_model"
    )
    assert snapshot["executor_descriptor"]["executor_id"] == (
        "autoregressive-draft"
    )
    assert snapshot["executor_descriptor"]["capabilities"] == (
        asdict(_FakeExecutor.capabilities)
    )
    assert snapshot["executor"]["rank"] == 0
    assert snapshot["executor"]["world_size"] == 1


def test_authority_snapshot_rejects_executor_topology_mismatch(
    tmp_path,
):
    runner = _runner(tmp_path)
    dependencies = _Dependencies()
    runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
    )
    dependencies.executor.rank = 1

    with pytest.raises(RuntimeError, match="topology mismatch"):
        runner.autoregressive_draft_authority_snapshot()


def test_fused_proposal_source_keeps_root_only_return_boundary():
    module = ast.parse(MODEL_RUNNER_PATH.read_text())
    model_runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "run_spec_first_target_and_proposal_batch"
    )
    selector_line = min(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "select_tensor_parallel_greedy_tokens"
    )
    execute_line = min(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute_batch"
    )
    nonroot_return_line = min(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(child, ast.Attribute)
            and child.attr == "rank"
            for child in ast.walk(node.test)
        )
        and any(
            isinstance(child, ast.Return)
            and isinstance(child.value, ast.Constant)
            and child.value.value is None
            for child in node.body
        )
    )
    fused_row_line = min(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FirstTargetProposalResult"
    )

    assert selector_line < execute_line
    assert execute_line < nonroot_return_line
    assert nonroot_return_line < fused_row_line


def test_generic_runtime_files_have_no_qwen3_draft_branch():
    forbidden = (
        "autoregressive_draft_model",
        "Qwen3AutoregressiveDraftBackend",
        "Qwen3DraftPhysicalSlotStore",
    )
    generic_paths = (
        ROOT / "tinyvllm/engine/llm_engine.py",
        ROOT / "tinyvllm/engine/scheduler.py",
        ROOT / "tinyvllm/speculative/verifier.py",
        ROOT / "tinyvllm/engine/speculative_proposal_executor.py",
        ROOT / "tinyvllm/engine/proposal_kv_cache.py",
        ROOT / "tinyvllm/engine/proposal_kv_lifecycle.py",
        ROOT / "tinyvllm/engine/speculative_side_state.py",
    )

    for path in generic_paths:
        source = path.read_text()
        assert not any(token in source for token in forbidden), path


def test_model_runner_exit_closes_draft_graph_before_destroying_process_group():
    tree = ast.parse(MODEL_RUNNER_PATH.read_text())
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    exit_method = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "exit"
    )
    close_lines = [
        node.lineno
        for node in ast.walk(exit_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
    ]
    destroy_line = next(
        node.lineno
        for node in ast.walk(exit_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "destroy_process_group"
    )

    assert close_lines
    assert max(close_lines) < destroy_line
