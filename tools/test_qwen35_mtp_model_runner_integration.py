from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"


@dataclass(frozen=True)
class _Descriptor:
    executor_id: str
    capabilities: object


def _load_runner_method(name):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(),
        filename=str(MODEL_RUNNER_PATH),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    method.decorator_list = []
    namespace = {
        "ModelRunnerProposalExecutorDescriptor": _Descriptor,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method], type_ignores=[])
            ),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class _Registry:

    def __init__(self, events):
        self.events = events
        self.rows = {}

    def register(self, executor_id, executor, capabilities):
        self.events.append("register")
        if executor_id in self.rows:
            raise ValueError("duplicate executor")
        self.rows[executor_id] = (executor, capabilities)


class _PhysicalStore:

    def __init__(self, events):
        self.events = events
        self.bound_backend = None

    def bind_attention_backend(self, backend):
        self.events.append("bind_attention_backend")
        self.bound_backend = backend


class _EntryAllocator:

    def __init__(self, storage):
        self.storage = storage

    def authority_snapshot(self):
        return {"allocator_mode": "fake"}


class _Dependencies:

    def __init__(self, events, *, fail_binding=False):
        self.events = events
        self.fail_binding = fail_binding
        self.read_sources = []
        self.physical_store = None
        self.entry_allocator = None
        self.allocator_kwargs = None
        self.build_topology = []
        self.executor_topology = []
        self.plan = SimpleNamespace(
            tensors=tuple(
                SimpleNamespace(source_name=f"mtp.source.{index}")
                for index in range(15)
            )
        )

    def read_metadata(self, _config):
        self.events.append("read_metadata")
        return SimpleNamespace(
            hf_config=SimpleNamespace(),
            index_payload=object(),
            shard_headers=object(),
        )

    def build_checkpoint_plan(
        self,
        hf_config,
        index_payload,
        shard_headers,
    ):
        assert hf_config is not None
        assert index_payload is not None
        assert shard_headers is not None
        self.events.append("build_plan")
        return self.plan

    def build_module(
        self,
        hf_config,
        *,
        embed_tokens,
        lm_head,
        tensor_parallel_size,
        tensor_parallel_rank,
    ):
        del hf_config
        self.build_topology.append((
            tensor_parallel_size,
            tensor_parallel_rank,
        ))
        self.events.append("build_module")
        attention_backend = SimpleNamespace()
        return SimpleNamespace(
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            layer=SimpleNamespace(
                decoder_layer=SimpleNamespace(
                    full_attention=SimpleNamespace(
                        attention_backend=attention_backend,
                    )
                )
            ),
        )

    def read_tensor(self, _config, tensor):
        self.read_sources.append(tensor.source_name)
        return tensor.source_name

    def bind_checkpoint(self, module, plan, tensor_reader):
        self.events.append("bind_checkpoint")
        if self.fail_binding:
            raise RuntimeError("injected MTP bind failure")
        assert module is not None
        return tuple(
            sorted(
                tensor_reader(tensor)
                for tensor in plan.tensors
            )
        )

    def move_module_to_device(self, module, target_model):
        assert module.embed_tokens is target_model.embed_tokens
        assert module.lm_head is target_model.lm_head
        self.events.append("move_module_to_device")

    def build_proposal_kv_allocator(self, config, module):
        assert module is not None
        self.events.append("build_proposal_kv_allocator")
        self.physical_store = _PhysicalStore(self.events)
        self.entry_allocator = _EntryAllocator(
            self.physical_store
        )
        self.allocator_kwargs = {
            "offload_enabled": (
                config.proposal_kv_offload_enabled
            ),
            "logical_entry_capacity": (
                config.proposal_kv_logical_entry_capacity
            ),
            "gpu_slot_capacity": (
                config.proposal_kv_gpu_slot_capacity
            ),
            "cpu_backing_capacity": (
                config.proposal_kv_cpu_backing_capacity
            ),
            "async_copy": config.proposal_kv_async_copy,
            "batch_copy": config.proposal_kv_batch_copy,
        }
        return self.entry_allocator

    def build_proposal_kv_cache(self, entry_allocator):
        assert entry_allocator is self.entry_allocator
        self.events.append("build_proposal_kv_cache")
        return SimpleNamespace(
            entry_allocator=entry_allocator,
            authority_snapshot=lambda: {
                "entry_allocator": (
                    entry_allocator.authority_snapshot()
                )
            },
        )

    def build_graph_runner(self, _config, _module, _cache):
        self.events.append("build_graph_runner")
        return None

    def build_executor(
        self,
        *,
        module,
        proposal_kv_cache,
        max_proposal_tokens,
        graph_runner,
        tensor_parallel_rank,
        tensor_parallel_size,
    ):
        assert module is not None
        assert proposal_kv_cache is not None
        assert max_proposal_tokens == 4
        assert graph_runner is None
        self.executor_topology.append((
            tensor_parallel_size,
            tensor_parallel_rank,
        ))
        self.events.append("build_executor")
        return SimpleNamespace(
            capabilities=SimpleNamespace(
                source_type="native_model_runner",
                max_proposal_tokens=4,
            )
        )


def _runner(
    *,
    enabled=True,
    tensor_parallel_size=1,
    kv_offload_mvp0=False,
    proposal_kv_offload_enabled=False,
    proposal_kv_logical_entry_capacity=0,
    proposal_kv_gpu_slot_capacity=0,
    proposal_kv_cpu_backing_capacity=0,
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    rank=0,
):
    events = ["target_loaded"]
    shared_weight = object()
    embed_tokens = SimpleNamespace(weight=shared_weight)
    lm_head = SimpleNamespace(weight=shared_weight)
    target_model = SimpleNamespace(
        embed_tokens=embed_tokens,
        lm_head=lm_head,
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            qwen35_mtp_enabled=enabled,
            qwen35_mtp_max_proposal_tokens=4,
            qwen35_mtp_cuda_graphs=False,
            proposal_kv_offload_enabled=(
                proposal_kv_offload_enabled
            ),
            proposal_kv_logical_entry_capacity=(
                proposal_kv_logical_entry_capacity
            ),
            proposal_kv_gpu_slot_capacity=(
                proposal_kv_gpu_slot_capacity
            ),
            proposal_kv_cpu_backing_capacity=(
                proposal_kv_cpu_backing_capacity
            ),
            proposal_kv_async_copy=True,
            proposal_kv_batch_copy=True,
            tensor_parallel_size=tensor_parallel_size,
            kv_offload_mvp0=kv_offload_mvp0,
            max_num_seqs=8,
            max_model_len=128,
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                text_config=SimpleNamespace(
                    mtp_num_hidden_layers=mtp_num_hidden_layers,
                    mtp_use_dedicated_embeddings=(
                        mtp_use_dedicated_embeddings
                    ),
                    tie_word_embeddings=True,
                ),
            ),
        ),
        rank=rank,
        world_size=tensor_parallel_size,
        model=target_model,
        qwen35_hybrid_model_owner=SimpleNamespace(
            model=target_model,
        ),
        speculative_proposal_executors=_Registry(events),
        qwen35_mtp_registration_error=None,
        events=events,
    )


def test_registration_occurs_after_target_load_and_exact_binding():
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    runner = _runner()
    dependencies = _Dependencies(runner.events)

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "native_checkpoint_proposal"
    assert runner.events == [
        "target_loaded",
        "read_metadata",
        "build_plan",
        "build_module",
        "bind_checkpoint",
        "move_module_to_device",
        "build_proposal_kv_allocator",
        "bind_attention_backend",
        "build_proposal_kv_cache",
        "build_graph_runner",
        "build_executor",
        "register",
    ]
    assert runner.qwen35_mtp_module.embed_tokens is (
        runner.model.embed_tokens
    )
    assert runner.qwen35_mtp_module.lm_head is runner.model.lm_head
    assert dependencies.read_sources == [
        f"mtp.source.{index}" for index in range(15)
    ]
    assert len(set(dependencies.read_sources)) == 15
    assert runner.qwen35_mtp_executor_descriptor is descriptor
    assert runner.qwen35_mtp_physical_store is (
        dependencies.physical_store
    )
    assert dependencies.physical_store.bound_backend is (
        runner.qwen35_mtp_module.layer.decoder_layer
        .full_attention.attention_backend
    )
    assert runner.qwen35_mtp_registration_error is None
    assert dependencies.build_topology == [(1, 0)]
    assert dependencies.executor_topology == [(1, 0)]
    assert dependencies.allocator_kwargs == {
        "offload_enabled": False,
        "logical_entry_capacity": 0,
        "gpu_slot_capacity": 0,
        "cpu_backing_capacity": 0,
        "async_copy": True,
        "batch_copy": True,
    }


@pytest.mark.parametrize("rank", range(4))
def test_tp4_registration_builds_rank_local_executor(rank):
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    runner = _runner(
        tensor_parallel_size=4,
        rank=rank,
    )
    dependencies = _Dependencies(runner.events)

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "native_checkpoint_proposal"
    assert dependencies.build_topology == [(4, rank)]
    assert dependencies.executor_topology == [(4, rank)]
    assert runner.qwen35_mtp_physical_store is (
        dependencies.physical_store
    )
    assert runner.qwen35_mtp_registration_error is None


def test_enabled_proposal_kv_offload_forwards_all_configuration():
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    runner = _runner(
        proposal_kv_offload_enabled=True,
        proposal_kv_logical_entry_capacity=16,
        proposal_kv_gpu_slot_capacity=4,
        proposal_kv_cpu_backing_capacity=16,
    )
    dependencies = _Dependencies(runner.events)

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "native_checkpoint_proposal"
    assert dependencies.allocator_kwargs == {
        "offload_enabled": True,
        "logical_entry_capacity": 16,
        "gpu_slot_capacity": 4,
        "cpu_backing_capacity": 16,
        "async_copy": True,
        "batch_copy": True,
    }


def test_target_kv_offload_does_not_disable_native_mtp_registration():
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    runner = _runner(kv_offload_mvp0=True)
    dependencies = _Dependencies(runner.events)

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == (
        "native_checkpoint_proposal"
    )
    assert runner.qwen35_mtp_executor_descriptor is descriptor
    assert runner.qwen35_mtp_registration_error is None


def test_default_registration_builds_on_cpu_then_moves_to_target_device():
    source = MODEL_RUNNER_PATH.read_text()
    start = source.index("def _qwen35_mtp_registration_dependencies():")
    end = source.index("\ndef _run_model_runner_eager(", start)
    registration_source = source[start:end]

    assert 'parameter_device="cpu"' in registration_source
    assert 'parameter_device="cuda"' not in registration_source
    assert "move_module_to_device=" in registration_source


def test_production_graph_builder_installs_exact_q_backend():
    source = MODEL_RUNNER_PATH.read_text()
    start = source.index("def _qwen35_mtp_registration_dependencies():")
    end = source.index("\ndef _run_model_runner_eager(", start)
    registration_source = source[start:end]

    assert (
        "Qwen3.5 MTP CUDA graph capture backend "
        "is not installed"
    ) not in registration_source
    assert "Qwen35MTPExactGraphRunner" in registration_source
    assert "Qwen35MTPCudaGraphBackend" in registration_source
    assert "Qwen35MTPGraphScratchOwner" in registration_source
    assert "scratch_cache = ProposalKVCache(" in registration_source
    assert "live_cache=_cache" in registration_source
    assert "scratch_cache=scratch_cache" in registration_source
    assert "block_table_width=int(config.max_model_len)" in (
        registration_source
    )
    assert "if not config.qwen35_mtp_cuda_graphs:" in (
        registration_source
    )
    assert "return None" in registration_source


@pytest.mark.parametrize(
    "runner",
    (
        _runner(enabled=False),
        _runner(tensor_parallel_size=2),
        _runner(mtp_num_hidden_layers=2),
        _runner(mtp_use_dedicated_embeddings=True),
    ),
)
def test_unsupported_or_disabled_configuration_does_not_register(
    runner,
):
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    dependencies = _Dependencies(runner.events)

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor is None
    assert runner.speculative_proposal_executors.rows == {}
    assert "register" not in runner.events


def test_failed_registration_leaves_target_model_usable():
    register = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )
    runner = _runner()
    target_model = runner.model
    dependencies = _Dependencies(
        runner.events,
        fail_binding=True,
    )

    descriptor = register(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor is None
    assert runner.model is target_model
    assert runner.speculative_proposal_executors.rows == {}
    assert runner.qwen35_mtp_registration_error == (
        "RuntimeError: injected MTP bind failure"
    )


def test_authority_snapshot_reports_rank_and_registration_state():
    snapshot_method = _load_runner_method(
        "qwen35_mtp_authority_snapshot"
    )
    baseline = SimpleNamespace(
        rank=2,
        world_size=4,
        qwen35_mtp_executor=None,
    )

    assert snapshot_method(baseline) == {
        "rank": 2,
        "world_size": 4,
        "registered": False,
        "executor": None,
    }

    native = SimpleNamespace(
        rank=3,
        world_size=4,
        config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(
                    num_attention_heads=16,
                    num_key_value_heads=4,
                )
            )
        ),
        model=SimpleNamespace(
            embed_tokens=object(),
            lm_head=object(),
        ),
        qwen35_mtp_module=SimpleNamespace(),
        qwen35_mtp_physical_store=SimpleNamespace(),
        qwen35_mtp_executor=SimpleNamespace(
            tp4_authority_snapshot=lambda: {
                "tensor_parallel_rank": 3,
                "tensor_parallel_size": 4,
                "active_transactions": 0,
            }
        ),
    )
    native.qwen35_mtp_module.embed_tokens = (
        native.model.embed_tokens
    )
    native.qwen35_mtp_module.lm_head = native.model.lm_head

    assert snapshot_method(native) == {
        "rank": 3,
        "world_size": 4,
        "registered": True,
        "module_type": "SimpleNamespace",
        "physical_store_type": "SimpleNamespace",
        "shared_embed_tokens": True,
        "shared_lm_head": True,
        "local_query_heads": 4,
        "local_kv_heads": 1,
        "executor": {
            "tensor_parallel_rank": 3,
            "tensor_parallel_size": 4,
            "active_transactions": 0,
        },
    }
