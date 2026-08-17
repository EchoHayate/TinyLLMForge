from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
from dataclasses import fields
import gc
import importlib.util
from itertools import count
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import weakref


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "tools"
    / "qwen35_constructed_engine_model_runner_ownership_preflight.py"
)


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "_qwen35_constructed_runtime_preflight_test",
        PREFLIGHT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(callback, message):
    try:
        callback()
    except (RuntimeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_source_contract_freezes_exact_constructor_identity_and_hashes():
    preflight = _load_preflight()

    contract = preflight.inspect_constructed_runtime_source_contract(ROOT)

    assert contract["files"] == {
        "tinyvllm/config.py": (
            "9b860eafe88c1734e5135ab0f65188f025762f5d0d0a702eb4994157aabec076"
        ),
        "tinyvllm/engine/llm_engine.py": (
            "6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae"
        ),
        "tinyvllm/engine/model_runner.py": (
            "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
        ),
    }
    assert contract["methods"] == {
        "LLMEngine.__init__": (
            "f770308d40248be4515838a720b288fd69f718d25746398bc145b4b43478fd9c"
        ),
        "LLMEngine.bind_qwen35_loaded_checkpoint_candidates": (
            "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c"
        ),
        "LLMEngine.call_model_runner_acknowledged": (
            "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d"
        ),
        "ModelRunner.__init__": (
            "8aa2747cff30e8398737cb024d375f9f04763efdd53cb23084c32c3d872f4edc"
        ),
        "ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate": (
            "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd"
        ),
        "ModelRunner.bind_qwen35_loaded_checkpoint_candidate": (
            "a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0"
        ),
        "ModelRunner.dispatch_command": (
            "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342"
        ),
        "ModelRunner.publish_qwen35_loaded_checkpoint_candidate": (
            "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
        ),
    }
    assert contract["constructor_signatures"] == {
        "LLMEngine.__init__": "(self, model, **kwargs)",
        "ModelRunner.__init__": (
            "(self, config: Config, rank: int, "
            "event: Event | list[Event], ack_sender=None)"
        ),
    }
    assert contract["forbidden_execution_forms"] == {
        "object_new": False,
        "constructor_ast_compile": False,
        "subclass_construction": False,
        "class_replacement": False,
    }


def test_source_contract_rejects_resigned_constructor():
    preflight = _load_preflight()
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "source"
        for relative in (
            "tinyvllm/config.py",
            "tinyvllm/engine/llm_engine.py",
            "tinyvllm/engine/model_runner.py",
        ):
            destination = copied / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        path = copied / "tinyvllm/engine/model_runner.py"
        path.write_text(path.read_text() + "\n")

        _expect_error(
            lambda: preflight.inspect_constructed_runtime_source_contract(
                copied
            ),
            "source hash mismatch",
        )


def test_ssh_command_preserves_remote_argument_boundaries():
    preflight = _load_preflight()
    script = "test ! -e /tmp/example && mkdir -p /tmp/example/source"

    command = preflight.build_ssh_command(["bash", "-c", script])

    assert command[:-1] == [
        "ssh",
        "-o",
        f"ControlPath={preflight.CONTROL_PATH}",
        preflight.REMOTE_TARGET,
    ]
    assert command[-1] == (
        "bash -c 'test ! -e /tmp/example && "
        "mkdir -p /tmp/example/source'"
    )


def test_source_closure_includes_frozen_constructor_sources():
    preflight = _load_preflight()
    oracle = (
        ROOT
        / "experiments/qwen35_hybrid_state/"
        "qwen35-tp4-real-candidate-replay-20260728-145713/"
        "tp4_real_candidate_provenance_oracle.json"
    )

    closure = preflight._source_closure_from_oracle(oracle)

    assert (
        set(preflight.EXPECTED_FILE_SHA256)
        | set(preflight.CONSTRUCTOR_RUNTIME_FILE_SHA256)
        | set(preflight.DIRECT_GATE_FILE_SHA256)
    ) <= set(closure)
    assert len(closure) == 63


def test_replacement_allowlist_is_exact_and_closed():
    preflight = _load_preflight()

    assert preflight.CONSTRUCTOR_REPLACEMENT_ALLOWLIST == {
        "llm_engine.Config",
        "llm_engine.ModelRunnerCommandAckCollector",
        "llm_engine.AutoTokenizer",
        "llm_engine.Scheduler",
        "llm_engine.mp.get_context",
        "llm_engine.atexit.register",
        "model_runner.dist.init_process_group",
        "model_runner.dist.barrier",
        "model_runner.torch.cuda.set_device",
        "model_runner.torch.get_default_dtype",
        "model_runner.torch.set_default_dtype",
        "model_runner.torch.set_default_device",
        "model_runner.set_quant_config",
        "model_runner.Qwen3ForCausalLM",
        "model_runner.load_model",
        "model_runner.apply_cpu_offload",
        "model_runner.Sampler",
        "model_runner.SharedMemory",
        "ModelRunner.warmup_model",
        "ModelRunner.allocate_kv_cache",
        "ModelRunner.capture_cudagraph",
        "ModelRunner.loop",
    }
    _expect_error(
        lambda: preflight.validate_constructor_replacement_names(
            preflight.CONSTRUCTOR_REPLACEMENT_ALLOWLIST
            | {"llm_engine.extra"}
        ),
        "replacement allowlist mismatch",
    )


def test_preflight_source_does_not_use_constructor_shortcuts():
    source = PREFLIGHT_PATH.read_text()
    tree = ast.parse(source, filename=str(PREFLIGHT_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert "object.__new__" not in source
    assert not any(
        isinstance(call.func, ast.Name)
        and call.func.id in {"compile", "exec"}
        for call in calls
    )
    assert "class ConstructedModelRunner" not in source
    assert "class ConstructedLLMEngine" not in source


def test_capsule_restores_every_replaced_identity_after_success():
    preflight = _load_preflight()

    targets = {
        "alpha.value": object(),
        "beta.value": object(),
    }
    replacements = {
        "alpha.value": object(),
        "beta.value": object(),
    }
    namespace = preflight.MutableDependencyNamespace(targets)
    original = namespace.identity_snapshot()

    with preflight.InertConstructorDependencyCapsule(
        namespace=namespace,
        replacements=replacements,
        allowed_names=set(replacements),
    ) as capsule:
        assert namespace.get("alpha.value") is replacements["alpha.value"]
        assert namespace.get("beta.value") is replacements["beta.value"]
        assert capsule.installed is True

    assert namespace.identity_snapshot() == original
    assert capsule.installed is False
    assert capsule.restoration_complete is True


def test_capsule_restores_every_replaced_identity_after_failure():
    preflight = _load_preflight()
    targets = {"alpha.value": object()}
    namespace = preflight.MutableDependencyNamespace(targets)
    original = namespace.identity_snapshot()

    try:
        with preflight.InertConstructorDependencyCapsule(
            namespace=namespace,
            replacements={"alpha.value": object()},
            allowed_names={"alpha.value"},
        ):
            raise RuntimeError("injected")
    except RuntimeError as error:
        assert str(error) == "injected"
    else:
        raise AssertionError("expected injected failure")

    assert namespace.identity_snapshot() == original


def test_capsule_rejects_unapproved_or_nested_installation():
    preflight = _load_preflight()
    namespace = preflight.MutableDependencyNamespace(
        {"alpha.value": object()}
    )
    capsule = preflight.InertConstructorDependencyCapsule(
        namespace=namespace,
        replacements={"alpha.value": object()},
        allowed_names={"alpha.value"},
    )

    with capsule:
        _expect_error(lambda: capsule.__enter__(), "already installed")

    _expect_error(
        lambda: preflight.InertConstructorDependencyCapsule(
            namespace=namespace,
            replacements={"alpha.value": object()},
            allowed_names={"beta.value"},
        ),
        "replacement allowlist mismatch",
    )


def test_module_dependency_namespace_replaces_nested_attributes():
    preflight = _load_preflight()
    original = object()
    replacement = object()
    module = types.SimpleNamespace(
        nested=types.SimpleNamespace(value=original)
    )
    namespace = preflight.ModuleDependencyNamespace(
        {"module": module}
    )

    assert namespace.get("module.nested.value") is original
    namespace.set("module.nested.value", replacement)
    assert module.nested.value is replacement
    namespace.set("module.nested.value", original)
    assert module.nested.value is original


def test_inert_tp4_config_has_constructor_safe_exact_fields():
    preflight = _load_preflight()
    dtype = object()

    config = preflight.build_inert_tp4_config(
        model="/approved/model",
        torch_dtype=dtype,
    )

    assert config.model == "/approved/model"
    assert config.tensor_parallel_size == 4
    assert config.enforce_eager is True
    assert config.cpu_offload is False
    assert config.kv_quant_bits == 0
    assert config.am_compact_blocks == 0
    assert config.kv_offload_mvp0 is False
    assert config.multi_sequence_cuda_graphs is False
    assert config.hf_config.torch_dtype is dtype
    assert config.kvcache_block_size == 256
    assert config.quantization is None
    assert config.act_quant_bits == 0
    assert config.smoothquant_scale_path is None


def test_inert_tp4_config_replacement_remains_a_dataclass_type():
    preflight = _load_preflight()
    ledger = preflight.DependencyLedger()
    dtype = object()

    config_class = preflight.build_inert_tp4_config_class(
        expected_model="/approved/model",
        torch_dtype=dtype,
        ledger=ledger,
    )

    assert isinstance(config_class, type)
    assert {field.name for field in fields(config_class)} >= {
        "model",
        "tensor_parallel_size",
        "enforce_eager",
        "hf_config",
    }
    config = config_class("/approved/model")
    assert config.tensor_parallel_size == 4
    assert config.hf_config.torch_dtype is dtype
    assert ledger.counts() == {"Config": 1}

    _expect_error(
        lambda: config_class("/different/model"),
        "model path changed",
    )


def test_temporary_module_registry_restores_existing_and_missing_entries():
    preflight = _load_preflight()
    existing_name = "_constructed_runtime_existing"
    missing_name = "_constructed_runtime_missing"
    existing = types.ModuleType(existing_name)
    replacement = types.ModuleType(existing_name)
    inserted = types.ModuleType(missing_name)
    sys.modules[existing_name] = existing
    sys.modules.pop(missing_name, None)

    try:
        with preflight.TemporaryModuleRegistry({
            existing_name: replacement,
            missing_name: inserted,
        }) as registry:
            assert sys.modules[existing_name] is replacement
            assert sys.modules[missing_name] is inserted
            assert registry.installed is True

        assert sys.modules[existing_name] is existing
        assert missing_name not in sys.modules
        assert registry.restoration_complete is True
    finally:
        sys.modules.pop(existing_name, None)
        sys.modules.pop(missing_name, None)


def test_remove_new_tinyvllm_modules_preserves_preexisting_entries():
    preflight = _load_preflight()
    existing_name = "tinyvllm._constructed_existing"
    new_name = "tinyvllm._constructed_new"
    existing = types.ModuleType(existing_name)
    inserted = types.ModuleType(new_name)
    sys.modules[existing_name] = existing
    before = set(sys.modules)
    sys.modules[new_name] = inserted

    try:
        removed = preflight.remove_new_tinyvllm_modules(before)

        assert new_name in removed
        assert new_name not in sys.modules
        assert sys.modules[existing_name] is existing
    finally:
        sys.modules.pop(existing_name, None)
        sys.modules.pop(new_name, None)


def test_constructed_transport_module_preserves_pickle_class_identity():
    preflight = _load_preflight()
    module_name = "tinyvllm.engine.model_runner_command_ack"
    transport_module = types.ModuleType(module_name)

    class Envelope:
        pass

    Envelope.__module__ = module_name
    Envelope.__qualname__ = "Envelope"
    transport_module.Envelope = Envelope
    transport_module.ModelRunnerCommandEnvelope = Envelope
    model_runner_module = types.SimpleNamespace(
        ModelRunnerCommandEnvelope=Envelope,
    )
    package_names = ("tinyvllm", "tinyvllm.engine")
    original_packages = {
        name: sys.modules.get(name) for name in package_names
    }
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    original = sys.modules.get(module_name)
    sys.modules[module_name] = transport_module
    before = set(sys.modules)
    before.remove(module_name)

    try:
        preserve = preflight.constructed_transport_module_preserve_names(
            model_runner_module=model_runner_module,
        )
        removed = preflight.remove_new_tinyvllm_modules(
            before,
            preserve=preserve,
        )

        assert preserve == (module_name,)
        assert module_name not in removed
        assert sys.modules[module_name] is transport_module
        assert pickle.loads(pickle.dumps(Envelope())).__class__ is Envelope
    finally:
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original
        for name, package in original_packages.items():
            if package is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = package


def test_constructed_transport_module_identity_is_restored_after_reload():
    preflight = _load_preflight()
    module_name = "tinyvllm.engine.model_runner_command_ack"
    canonical = types.ModuleType(module_name)

    class Envelope:
        pass

    Envelope.__module__ = module_name
    Envelope.__qualname__ = "ModelRunnerCommandEnvelope"
    canonical.ModelRunnerCommandEnvelope = Envelope
    model_runner_module = types.SimpleNamespace(
        ModelRunnerCommandEnvelope=Envelope,
    )
    replacement = types.ModuleType(module_name)
    replacement.ModelRunnerCommandEnvelope = type(
        "ModelRunnerCommandEnvelope",
        (),
        {"__module__": module_name},
    )
    package_names = ("tinyvllm", "tinyvllm.engine")
    originals = {
        name: sys.modules.get(name)
        for name in (*package_names, module_name)
    }
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules[module_name] = replacement

    try:
        evidence = (
            preflight.restore_constructed_transport_module_identity(
                model_runner_module=model_runner_module,
                transport_module=canonical,
            )
        )

        assert evidence == {
            "module_name": module_name,
            "restored": True,
            "envelope_class_identity": True,
        }
        assert sys.modules[module_name] is canonical
        assert (
            pickle.loads(pickle.dumps(Envelope())).__class__
            is Envelope
        )
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_import_time_stub_module_set_is_exact():
    preflight = _load_preflight()

    assert preflight.IMPORT_TIME_STUB_MODULES == {
        "tinyvllm.engine.flash_attn_split_policy",
        "tinyvllm.engine.kv_cartridge",
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication",
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore",
        "tinyvllm.engine.qwen35_hybrid_prefix_owner",
        "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
        "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
        "tinyvllm.engine.qwen35_hybrid_prefix_source_publication",
        "tinyvllm.engine.scheduler",
        "tinyvllm.engine.sequence",
        "tinyvllm.layers.linear",
        "tinyvllm.layers.sampler",
        "tinyvllm.models.qwen3",
        "tinyvllm.sampling_params",
        "tinyvllm.utils.cpu_offload",
        "tinyvllm.utils.loader",
    }


def test_import_time_stubs_cover_production_import_surface():
    preflight = _load_preflight()
    stubs = preflight.build_import_time_stubs()
    required = {
        "tinyvllm.engine.flash_attn_split_policy": {
            "FlashAttentionSplitInputs",
            "build_flash_attn_263_graph_identity",
        },
        "tinyvllm.engine.kv_cartridge": {
            "compress_decode_block_table_rows",
            "should_use_kv_cartridge",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication": {
            "Qwen35HybridPrefixEnginePublicationCoordinator",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore": {
            "Qwen35HybridPrefixEngineRestoreCoordinator",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_owner": {
            "Qwen35HybridPrefixRestoreOwner",
            "build_qwen35_hybrid_prefix_restore_owner",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket": {
            "Qwen35HybridPrefixPublicationParticipant",
            "Qwen35HybridPrefixPublicationPayload",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket": {
            "Qwen35HybridPrefixRestoreParticipant",
        },
        "tinyvllm.engine.qwen35_hybrid_prefix_source_publication": {
            "Qwen35HybridPrefixSourcePublisher",
        },
        "tinyvllm.engine.scheduler": {"Scheduler"},
        "tinyvllm.engine.sequence": {"Sequence"},
        "tinyvllm.sampling_params": {"SamplingParams"},
    }

    for module_name, names in required.items():
        assert names <= set(vars(stubs[module_name]))


def test_import_time_linear_stub_exports_checkpoint_type_surface():
    preflight = _load_preflight()
    stubs = preflight.build_import_time_stubs()
    linear = stubs["tinyvllm.layers.linear"]

    for name in (
        "LinearBase",
        "ColumnParallelLinear",
        "HeadPairedColumnParallelLinear",
        "KVHeadParallelLinear",
        "MergedColumnParallelLinear",
        "QKVParallelLinear",
        "ReplicatedHeadPairedColumnParallelLinear",
        "ReplicatedKVHeadParallelLinear",
        "ReplicatedLinear",
        "ReplicatedLocalOutputLinear",
        "ReplicatedMergedColumnParallelLinear",
        "ReplicatedWeightRowParallelLinear",
        "RowParallelLinear",
        "SegmentedColumnParallelLinear",
    ):
        value = getattr(linear, name)
        assert isinstance(value, type)
    assert callable(linear.get_quant_method)
    assert callable(linear.set_quant_config)


def test_inert_spawn_context_defers_exact_runner_targets():
    preflight = _load_preflight()
    constructed = []

    def target(config, rank, event, sender):
        runner = types.SimpleNamespace(
            config=config,
            rank=rank,
            event=event,
            sender=sender,
        )
        constructed.append(runner)
        return runner

    ledger = preflight.DependencyLedger()
    context = preflight.InertSpawnContext(ledger=ledger)
    receiver, sender = context.Pipe(duplex=False)
    event = context.Event()
    process = context.Process(
        target=target,
        args=("config", 1, event, sender),
    )

    process.start()

    assert constructed == []
    assert process.started is True
    assert process.is_alive() is True
    assert process.target is target
    assert process.args == ("config", 1, event, sender)
    assert receiver.rank is None
    assert sender.closed is False

    runner = process.construct_deferred()

    assert runner is constructed[0]
    assert process.runner is runner
    assert process.construct_deferred() is runner
    assert ledger.counts() == {
        "context.Event": 1,
        "context.Pipe": 1,
        "context.Process": 1,
        "process.start": 1,
        "runner.deferred_construct": 1,
    }


def test_inert_shared_memory_uses_private_registry_and_closes():
    preflight = _load_preflight()
    ledger = preflight.DependencyLedger()
    registry = preflight.InertSharedMemoryRegistry(ledger=ledger)

    owner = registry.open(name="tinyvllm", create=True, size=2**20)
    attached = registry.open(name="tinyvllm")

    assert owner is not attached
    assert owner.buf is attached.buf
    assert len(owner.buf) == 2**20
    owner.buf[:4] = b"test"
    assert bytes(attached.buf[:4]) == b"test"
    owner.close()
    attached.close()
    owner.unlink()

    assert registry.resource_names() == ()
    assert ledger.counts() == {
        "SharedMemory.attach": 1,
        "SharedMemory.close": 2,
        "SharedMemory.create": 1,
        "SharedMemory.unlink": 1,
    }


@dataclass(frozen=True)
class _Envelope:
    command_id: int
    method_name: str
    args: tuple
    requires_ack: bool


@dataclass(frozen=True)
class _Ack:
    command_id: int
    rank: int
    status: str
    result: object


def test_in_process_ack_collector_constructs_workers_and_collects_empty_payload():
    preflight = _load_preflight()
    ledger = preflight.DependencyLedger()
    context = preflight.InertSpawnContext(ledger=ledger)
    processes = []

    def target(config, rank, event, sender):
        return types.SimpleNamespace(
            rank=rank,
            bind_published_qwen35_loaded_checkpoint_candidate=(
                lambda rank=rank: {"participant_id": rank}
            ),
        )

    receivers = []
    for rank in range(1, 4):
        receiver, sender = context.Pipe(duplex=False)
        receiver.rank = rank
        process = context.Process(
            target=target,
            args=("config", rank, context.Event(), sender),
        )
        process.start()
        processes.append(process)
        receivers.append((rank, receiver))

    envelope = _Envelope(
        command_id=7,
        method_name=(
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        args=(),
        requires_ack=True,
    )
    collector = preflight.InProcessAckCollector(
        tuple(receivers),
        processes=tuple(processes),
        envelope_reader=lambda: envelope,
        ack_factory=_Ack,
        ledger=ledger,
    )

    assert [process.runner.rank for process in processes] == [1, 2, 3]
    acknowledgements = collector.collect(
        7,
        expected_ranks=(1, 2, 3),
        timeout_s=0.25,
        is_rank_alive=lambda rank: processes[rank - 1].is_alive(),
    )

    assert acknowledgements == (
        _Ack(7, 1, "ok", {"participant_id": 1}),
        _Ack(7, 2, "ok", {"participant_id": 2}),
        _Ack(7, 3, "ok", {"participant_id": 3}),
    )
    assert collector.collect_calls == 1
    assert collector.last_envelope is envelope
    assert ledger.counts()["ack_collector.construct"] == 1
    assert ledger.counts()["ack_collector.collect"] == 1
    assert ledger.counts()["worker.bind.invoke"] == 3


def test_in_process_ack_collector_rejects_non_empty_payload():
    preflight = _load_preflight()
    ledger = preflight.DependencyLedger()
    context = preflight.InertSpawnContext(ledger=ledger)
    receiver, sender = context.Pipe(duplex=False)
    receiver.rank = 1
    process = context.Process(
        target=lambda *args: types.SimpleNamespace(rank=1),
        args=(None, 1, context.Event(), sender),
    )
    process.start()
    collector = preflight.InProcessAckCollector(
        ((1, receiver),),
        processes=(process,),
        envelope_reader=lambda: _Envelope(
            command_id=0,
            method_name=(
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            args=("candidate",),
            requires_ack=True,
        ),
        ack_factory=_Ack,
        ledger=ledger,
    )

    _expect_error(
        lambda: collector.collect(
            0,
            expected_ranks=(1,),
            timeout_s=0.25,
            is_rank_alive=lambda rank: True,
        ),
        "zero-payload",
    )


def test_construct_runtime_calls_exact_engine_and_runner_classes():
    preflight = _load_preflight()
    original_runner_class = None

    class FakeRunner:
        constructor_ranks = []

        def __init__(self, config, rank, event, ack_sender=None):
            FakeRunner.constructor_ranks.append(rank)
            self.config = config
            self.rank = rank
            self.world_size = config.tensor_parallel_size
            self.event = event
            self.ack_sender = ack_sender
            fake_model_runner.dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:2333",
                world_size=self.world_size,
                rank=rank,
            )
            fake_model_runner.torch.cuda.set_device(rank)
            fake_model_runner.Qwen3ForCausalLM(config.hf_config)
            fake_model_runner.load_model(None, config.model)
            fake_model_runner.Sampler()
            self.warmup_model()
            self.allocate_kv_cache()
            if self.world_size > 1:
                if rank == 0:
                    self.shm = fake_model_runner.SharedMemory(
                        name="tinyvllm",
                        create=True,
                        size=2**20,
                    )
                    fake_model_runner.dist.barrier()
                else:
                    fake_model_runner.dist.barrier()
                    self.shm = fake_model_runner.SharedMemory(
                        name="tinyvllm"
                    )
                    self.loop()

        def warmup_model(self):
            raise AssertionError("real warmup called")

        def allocate_kv_cache(self):
            raise AssertionError("real KV allocation called")

        def capture_cudagraph(self):
            raise AssertionError("real CUDA graph called")

        def loop(self):
            raise AssertionError("real worker loop called")

    original_runner_class = FakeRunner

    class FakeEngine:
        constructor_calls = 0

        def __init__(self, model, **kwargs):
            FakeEngine.constructor_calls += 1
            config = fake_llm_engine.Config(model, **kwargs)
            self.ps = []
            self.events = []
            context = fake_llm_engine.mp.get_context("spawn")
            receivers = []
            senders = []
            for rank in range(1, config.tensor_parallel_size):
                receiver, sender = context.Pipe(duplex=False)
                receiver.rank = rank
                receivers.append((rank, receiver))
                senders.append((rank, sender))
            sender_by_rank = dict(senders)
            for rank in range(1, config.tensor_parallel_size):
                event = context.Event()
                process = context.Process(
                    target=fake_llm_engine.ModelRunner,
                    args=(config, rank, event, sender_by_rank[rank]),
                )
                process.start()
                sender_by_rank[rank].close()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = fake_llm_engine.ModelRunner(
                config,
                0,
                self.events,
            )
            self.model_runner_ack_collector = (
                fake_llm_engine.ModelRunnerCommandAckCollector(
                    tuple(receivers)
                )
            )
            self.tokenizer = (
                fake_llm_engine.AutoTokenizer.from_pretrained(
                    config.model,
                    use_fast=True,
                )
            )
            self.scheduler = fake_llm_engine.Scheduler(config)
            fake_llm_engine.atexit.register(self.exit)

        def exit(self):
            raise AssertionError("exit must not be registered")

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        float32=object(),
        cuda=types.SimpleNamespace(
            set_device=lambda rank: None,
        ),
        get_default_dtype=lambda: object(),
        set_default_dtype=lambda value: None,
        set_default_device=lambda value: None,
    )
    fake_model_runner = types.SimpleNamespace(
        dist=types.SimpleNamespace(
            init_process_group=lambda **kwargs: None,
            barrier=lambda: None,
        ),
        torch=fake_torch,
        set_quant_config=lambda *args: None,
        Qwen3ForCausalLM=lambda config: object(),
        load_model=lambda *args, **kwargs: None,
        apply_cpu_offload=lambda *args, **kwargs: None,
        Sampler=lambda: object(),
        SharedMemory=lambda **kwargs: object(),
        ModelRunner=FakeRunner,
    )
    fake_llm_engine = types.SimpleNamespace(
        Config=object(),
        ModelRunner=FakeRunner,
        ModelRunnerCommandAckCollector=lambda receivers: object(),
        AutoTokenizer=object(),
        Scheduler=lambda config: object(),
        mp=types.SimpleNamespace(get_context=lambda name: None),
        atexit=types.SimpleNamespace(register=lambda callback: None),
        LLMEngine=FakeEngine,
    )
    original_identities = {
        "engine_class": id(fake_llm_engine.LLMEngine),
        "runner_class": id(fake_llm_engine.ModelRunner),
        "runner_module_class": id(fake_model_runner.ModelRunner),
    }

    scope = preflight.construct_engine_runtime_under_inert_capsule(
        llm_engine_module=fake_llm_engine,
        model_runner_module=fake_model_runner,
        torch_module=fake_torch,
        model="/approved/model",
    )

    assert type(scope.engine) is FakeEngine
    assert [type(scope.runners_by_rank[rank]) for rank in range(4)] == [
        original_runner_class,
        original_runner_class,
        original_runner_class,
        original_runner_class,
    ]
    assert FakeEngine.constructor_calls == 1
    assert FakeRunner.constructor_ranks == [0, 1, 2, 3]
    assert scope.constructor_evidence()["engine_constructor_count"] == 1
    assert scope.constructor_evidence()["runner_constructor_ranks"] == [
        0,
        1,
        2,
        3,
    ]
    assert id(fake_llm_engine.LLMEngine) == original_identities["engine_class"]
    assert id(fake_llm_engine.ModelRunner) == original_identities["runner_class"]
    assert (
        id(fake_model_runner.ModelRunner)
        == original_identities["runner_module_class"]
    )
    assert scope.restoration_complete is True


def test_constructor_evidence_requires_exact_call_counts():
    preflight = _load_preflight()
    counts = dict(preflight.EXPECTED_CONSTRUCTOR_CALL_COUNTS)
    evidence = {
        "engine_constructor_count": 1,
        "runner_constructor_count": 4,
        "runner_constructor_ranks": [0, 1, 2, 3],
        "dependency_call_counts": counts,
        "restoration_complete": True,
    }

    assert (
        preflight.validate_constructor_evidence(evidence)
        is evidence
    )

    for name in (
        "dist.init_process_group",
        "ModelRunner.loop",
        "context.Process",
        "Scheduler",
    ):
        damaged = {
            **evidence,
            "dependency_call_counts": {
                **counts,
                name: counts[name] + 1,
            },
        }
        _expect_error(
            lambda damaged=damaged: (
                preflight.validate_constructor_evidence(damaged)
            ),
            "constructor call counts",
        )


def test_transfer_candidate_to_constructed_runner_publishes_without_binding():
    preflight = _load_preflight()

    class Runner:
        def __init__(self):
            self.rank = 2
            self.world_size = 4
            self.model = object()
            self.qwen35_loaded_checkpoint_candidate_slot = (
                types.SimpleNamespace(candidate=None)
            )
            self.publish_calls = []
            self.bind_calls = 0

        def publish_qwen35_loaded_checkpoint_candidate(self, candidate):
            self.publish_calls.append(candidate)
            self.qwen35_loaded_checkpoint_candidate_slot.candidate = (
                candidate
            )
            return candidate

        def bind_qwen35_loaded_checkpoint_candidate(self, candidate):
            self.bind_calls += 1
            raise AssertionError("candidate bound before Engine dispatch")

    runner = Runner()
    model = object()
    owner = types.SimpleNamespace(model=model)
    candidate = types.SimpleNamespace(owner=owner)

    evidence = preflight.transfer_candidate_to_constructed_runner(
        runner=runner,
        expected_runner_type=Runner,
        candidate=candidate,
        expected_rank=2,
    )

    assert runner.model is model
    assert runner.publish_calls == [candidate]
    assert runner.bind_calls == 0
    assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is (
        candidate
    )
    assert evidence == {
        "rank": 2,
        "exact_runner_class": True,
        "world_size": 4,
        "constructor_placeholder_replaced": True,
        "candidate_published": True,
        "candidate_bound_before_engine_dispatch": False,
    }


def test_transfer_candidate_rejects_wrong_rank_or_class():
    preflight = _load_preflight()

    class Runner:
        rank = 1
        world_size = 4
        model = object()
        qwen35_loaded_checkpoint_candidate_slot = types.SimpleNamespace(
            candidate=None
        )

        def publish_qwen35_loaded_checkpoint_candidate(self, candidate):
            self.qwen35_loaded_checkpoint_candidate_slot.candidate = (
                candidate
            )

    candidate = types.SimpleNamespace(
        owner=types.SimpleNamespace(model=object())
    )

    _expect_error(
        lambda: preflight.transfer_candidate_to_constructed_runner(
            runner=Runner(),
            expected_runner_type=Runner,
            candidate=candidate,
            expected_rank=2,
        ),
        "rank",
    )
    _expect_error(
        lambda: preflight.transfer_candidate_to_constructed_runner(
            runner=types.SimpleNamespace(rank=1, world_size=4),
            expected_runner_type=Runner,
            candidate=candidate,
            expected_rank=1,
        ),
        "class",
    )


def test_rebind_constructed_runner_candidate_types_updates_exact_globals():
    preflight = _load_preflight()
    old_candidate = type("OldCandidate", (), {})
    old_owner = type("OldOwner", (), {})
    old_slot = type("OldSlot", (), {})
    old_identity = lambda *_args: None
    module = types.SimpleNamespace(
        Qwen35LoadedCheckpointCandidate=old_candidate,
        Qwen35HybridModelOwner=old_owner,
        Qwen35HybridModelOwnerPublicationSlot=old_slot,
        _bind_qwen35_hybrid_prefix_runtime_identity=old_identity,
    )
    candidate = type("Candidate", (), {})
    owner = type("Owner", (), {})
    slot = type("Slot", (), {})
    identity = lambda *_args: None

    evidence = preflight.rebind_constructed_runner_candidate_types(
        model_runner_module=module,
        candidate_type=candidate,
        owner_type=owner,
        publication_slot_type=slot,
        identity_binder=identity,
    )

    assert module.Qwen35LoadedCheckpointCandidate is candidate
    assert module.Qwen35HybridModelOwner is owner
    assert module.Qwen35HybridModelOwnerPublicationSlot is slot
    assert (
        module._bind_qwen35_hybrid_prefix_runtime_identity
        is identity
    )
    assert evidence["candidate_type_rebound"] is True
    assert evidence["owner_type_rebound"] is True
    assert evidence["publication_slot_type_rebound"] is True
    assert evidence["identity_binder_rebound"] is True


def test_prepare_candidate_uses_proven_runtime_once_and_publishes():
    preflight = _load_preflight()
    model = object()
    owner = types.SimpleNamespace(model=model)
    candidate = types.SimpleNamespace(
        owner=owner,
        stats=types.SimpleNamespace(
            assigned_bindings=320,
            source_tensors=320,
            shard_count=1,
            loaded_bytes=3763655360,
            peak_source_bytes=1017118720,
        ),
    )
    target = types.SimpleNamespace()
    request = object()
    calls = {
        "factory": 0,
        "loader": 0,
        "validator": 0,
        "payload": 0,
    }

    def private_graph_factory():
        calls["factory"] += 1

        def loader(value):
            assert value is request
            calls["loader"] += 1
            return candidate

        return target, request, loader

    def candidate_validator(**kwargs):
        assert kwargs == {
            "candidate": candidate,
            "target": target,
            "model_fingerprint": "f" * 64,
        }
        calls["validator"] += 1
        return {"loaded_state_verified": True}

    def payload_recorder(**kwargs):
        assert kwargs == {
            "candidate": candidate,
            "target": target,
            "model_fingerprint": "f" * 64,
        }
        calls["payload"] += 1
        return {
            "binding_hash_count": 320,
            "binding_destination_sha256": ["a" * 64] * 320,
            "phase_hash_count": 26,
            "phase_destination_sha256": {
                f"phase-{index}": "b" * 64
                for index in range(26)
            },
            "aggregate_destination_sha256": "c" * 64,
            "alias_groups": [["x", "y"]],
        }

    class Runner:
        rank = 0
        world_size = 4

        def __init__(self):
            self.model = object()
            self.qwen35_loaded_checkpoint_candidate_slot = (
                types.SimpleNamespace(candidate=None)
            )

        def publish_qwen35_loaded_checkpoint_candidate(self, value):
            self.qwen35_loaded_checkpoint_candidate_slot.candidate = value
            return value

    state = preflight.prepare_candidate_for_constructed_runner(
        runner=Runner(),
        expected_runner_type=Runner,
        rank=0,
        model_fingerprint="f" * 64,
        scope_kwargs={
            "private_graph_factory": private_graph_factory,
            "candidate_validator": candidate_validator,
            "payload_recorder": payload_recorder,
        },
        pristine_row=None,
        pristine_validator=None,
    )

    assert calls == {
        "factory": 1,
        "loader": 1,
        "validator": 1,
        "payload": 1,
    }
    assert state.candidate is candidate
    assert state.target is target
    assert state.runner.model is model
    assert state.payload["binding_hash_count"] == 320
    assert state.transfer_evidence["candidate_published"] is True


def test_prebind_pristine_validator_does_not_require_method_execution():
    preflight = _load_preflight()
    payload = {
        "binding_destination_sha256": ["a" * 64],
        "phase_destination_sha256": {"phase": "b" * 64},
        "aggregate_destination_sha256": "c" * 64,
        "alias_groups": [["x", "y"]],
        "loader_stats": {"loaded_bytes": 1},
        "anticipated_identity": {
            "model_fingerprint": "f" * 64,
            "layout_fingerprint": "l" * 64,
            "dtype": "bfloat16",
        },
    }
    pristine = {
        "tp_rank": 2,
        "binding_destination_sha256": ["a" * 64],
        "phase_destination_sha256": {"phase": "b" * 64},
        "aggregate_destination_sha256": "c" * 64,
        "alias_groups": [["x", "y"]],
        "loader_stats": {"loaded_bytes": 1},
        "model_manifest_sha256": "f" * 64,
        "layout_fingerprint": "l" * 64,
        "dtype": "bfloat16",
    }

    assert preflight.validate_prebind_payload_against_pristine_rank(
        payload,
        pristine,
        rank=2,
    ) is payload

    assert "method_row" not in payload


def test_bind_constructed_runtime_uses_zero_payload_and_exact_repeat():
    preflight = _load_preflight()
    rows = tuple(
        {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": "f" * 64,
            "layout_fingerprint": "l" * 64,
            "dtype": "bfloat16",
            "detail": "",
        }
        for rank in range(4)
    )

    class Collector:
        collect_calls = 1
        last_envelope = _Envelope(
            command_id=0,
            method_name=(
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            args=(),
            requires_ack=True,
        )
        last_acknowledgements = tuple(
            types.SimpleNamespace(
                command_id=0,
                rank=rank,
                status="ok",
                result=rows[rank],
                error_type="",
                error_detail="",
            )
            for rank in (1, 2, 3)
        )

    class Engine:
        def __init__(self):
            self.calls = 0
            self.model_runner_ack_collector = Collector()
            self.qwen35_loaded_checkpoint_candidate_binding_rows = None
            self.qwen35_loaded_checkpoint_candidate_binding_configuration = (
                None
            )

        def bind_qwen35_loaded_checkpoint_candidates(self, *, timeout_s):
            self.calls += 1
            if (
                self.qwen35_loaded_checkpoint_candidate_binding_rows
                is None
            ):
                self.qwen35_loaded_checkpoint_candidate_binding_rows = rows
                self.qwen35_loaded_checkpoint_candidate_binding_configuration = (
                    "f" * 64,
                    "l" * 64,
                    "bfloat16",
                    float(timeout_s),
                )
            return self.qwen35_loaded_checkpoint_candidate_binding_rows

    engine = Engine()
    first = preflight.bind_constructed_runtime_candidates(
        engine=engine,
        expected_engine_type=Engine,
        timeout_s=0.25,
    )
    second = preflight.bind_constructed_runtime_candidates(
        engine=engine,
        expected_engine_type=Engine,
        timeout_s=0.25,
    )

    assert first["rows"] == rows
    assert second["rows"] is first["rows"]
    assert first["zero_payload_command"] is True
    assert first["exact_repeat_zero_dispatch"] is None
    assert second["exact_repeat_zero_dispatch"] is True
    assert engine.calls == 2


def test_release_constructed_runtime_clears_reverse_rank_graph():
    preflight = _load_preflight()
    clear_order = []

    class Tensor:
        def __init__(self, rank):
            self.rank = rank
            self.value = 1

        def zero_(self):
            clear_order.append(self.rank)
            self.value = 0

        def count_nonzero(self):
            return types.SimpleNamespace(item=lambda: self.value)

    class Slot:
        def __init__(self, candidate):
            self._publication = (candidate, candidate.owner, "f" * 64)

    class Runner:
        pass

    class Box:
        pass

    class Scope:
        released = False

        def close_inert_resources(self):
            self.resources_closed = True

    def build_live_scope():
        scope = Scope()
        scope.resources_closed = False
        engine = Box()
        collector = Box()
        collector.last_envelope = object()
        engine.model_runner_ack_collector = collector
        engine.model_runner = Box()
        engine.scheduler = Box()
        engine.tokenizer = Box()
        scope.engine = engine
        scope.runners_by_rank = {}
        scope.context = types.SimpleNamespace(processes=[])
        scope.shared_memory_registry = types.SimpleNamespace(
            resource_names=lambda: (),
        )
        states = []
        references = {}
        for rank in range(4):
            model = Box()
            model.rank = rank
            runtime_bridge = Box()
            runtime_bridge.rank = rank
            pool = Box()
            pool.rank = rank
            owner = Box()
            owner.model = model
            owner.runtime_bridge = runtime_bridge
            owner.pool = pool
            candidate = Box()
            candidate.owner = owner
            runner = Runner()
            runner.rank = rank
            runner.model = model
            runner.qwen35_loaded_checkpoint_candidate_slot = Slot(candidate)
            runner.qwen35_hybrid_model_owner = owner
            runner.hybrid_state_runtime_bridge = owner.runtime_bridge
            runtime_identity = Box()
            runtime_identity.rank = rank
            runner.qwen35_hybrid_prefix_runtime_identity = runtime_identity
            runner.qwen35_hybrid_prefix_runtime_identity_owner = owner
            tensor = Tensor(rank)
            target = Box()
            target.rank = rank
            request = Box()
            request.rank = rank
            state = types.SimpleNamespace(
                rank=rank,
                runner=runner,
                target=target,
                request=request,
                candidate=candidate,
                selected={rank: tensor},
                selected_ids={id(tensor)},
                registered=(tensor,),
                identity_snapshot=(id(tensor),),
                non_selected_values={},
                pool=owner.pool,
                pool_snapshot=("pool", rank),
                require_identity_unchanged=lambda values, snapshot: None,
                pool_unchanged=lambda pool, snapshot: True,
                no_grad=lambda: contextlib.nullcontext(),
            )
            scope.runners_by_rank[rank] = runner
            states.append(state)
            references[f"runner_{rank}"] = weakref.ref(runner)
            references[f"candidate_{rank}"] = weakref.ref(candidate)
            references[f"owner_{rank}"] = weakref.ref(owner)
        return {
            "constructed_scope": scope,
            "rank_states": states,
        }, references

    live_scope, references = build_live_scope()
    result = preflight.release_constructed_runtime(live_scope)
    gc.collect()

    assert result["release_rank_order"] == [3, 2, 1, 0]
    assert clear_order == [3, 2, 1, 0]
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["non_selected_tensors_unchanged"] is True
    assert result["tensor_identity_preserved"] is True
    assert result["pool_unchanged"] is True
    assert result["all_inert_resources_closed"] is True
    assert result["production_exit_call_count"] == 0
    assert result["all_private_objects_collected"] is True
    assert live_scope == {}
    assert all(reference() is None for reference in references.values())


def test_build_constructed_runtime_artifact_requires_complete_gate():
    preflight = _load_preflight()
    rows = [
        {
            "rank": rank,
            "binding_hash_count": 320,
            "binding_destination_sha256": ["a" * 64] * 320,
            "phase_hash_count": 26,
            "phase_destination_sha256": {
                f"phase_{index}": "b" * 64 for index in range(26)
            },
            "aggregate_destination_sha256": "c" * 64,
            "alias_groups": [[index, index + 1] for index in range(24)],
            "loader_stats": {"assigned_bindings": 320},
            "anticipated_identity": {
                "model_fingerprint": "f" * 64,
                "layout_fingerprint": "l" * 64,
                "dtype": "bfloat16",
            },
            "transfer_evidence": {
                "candidate_published": True,
                "candidate_bound_before_engine_dispatch": False,
            },
        }
        for rank in range(4)
    ]
    bound_rows = tuple(
        {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": "f" * 64,
            "layout_fingerprint": "l" * 64,
            "dtype": "bfloat16",
            "detail": "",
        }
        for rank in range(4)
    )
    smoke = {
        "status": "PASS",
        "source_contract": {
            "files": {"tinyvllm/config.py": "d" * 64},
            "methods": {"LLMEngine.__init__": "e" * 64},
            "constructor_signatures": {
                "LLMEngine.__init__": "(self, model, **kwargs)",
            },
            "forbidden_execution_forms": {
                "object_new": False,
                "constructor_ast_compile": False,
                "subclass_construction": False,
                "class_replacement": False,
            },
        },
        "constructor_evidence": {
            "engine_constructor_count": 1,
            "runner_constructor_count": 4,
            "runner_constructor_ranks": [0, 1, 2, 3],
            "dependency_call_counts": dict(
                preflight.EXPECTED_CONSTRUCTOR_CALL_COUNTS
            ),
            "original_dependency_identities": {"x": 1},
            "restored_dependency_identities": {"x": 1},
            "restoration_complete": True,
        },
        "class_identity": {
            "engine_exact_class": True,
            "runner_exact_class_by_rank": [True, True, True, True],
        },
        "constructor_ledger": [
            {
                "sequence": 0,
                "dependency": "LLMEngine.__init__",
                "rank": None,
                "arguments": {},
                "result_identity": "builtins.type:1",
            }
        ],
        "rank_payloads": rows,
        "first_binding": {
            "rows": bound_rows,
            "configuration": (
                "f" * 64,
                "l" * 64,
                "bfloat16",
                0.25,
            ),
            "command_envelope": {
                "command_id": 0,
                "method_name": (
                    "bind_published_qwen35_loaded_checkpoint_candidate"
                ),
                "args": [],
                "requires_ack": True,
            },
            "worker_acknowledgements": [
                {
                    "command_id": 0,
                    "rank": rank,
                    "status": "ok",
                    "result": bound_rows[rank],
                    "error_type": "",
                    "error_detail": "",
                }
                for rank in (1, 2, 3)
            ],
            "zero_payload_command": True,
            "exact_repeat_zero_dispatch": None,
        },
        "repeat_binding": {
            "rows": bound_rows,
            "configuration": (
                "f" * 64,
                "l" * 64,
                "bfloat16",
                0.25,
            ),
            "command_envelope": {
                "command_id": 0,
                "method_name": (
                    "bind_published_qwen35_loaded_checkpoint_candidate"
                ),
                "args": [],
                "requires_ack": True,
            },
            "worker_acknowledgements": [
                {
                    "command_id": 0,
                    "rank": rank,
                    "status": "ok",
                    "result": bound_rows[rank],
                    "error_type": "",
                    "error_detail": "",
                }
                for rank in (1, 2, 3)
            ],
            "zero_payload_command": True,
            "exact_repeat_zero_dispatch": True,
        },
        "transport_restoration": {
            "module_name": "tinyvllm.engine.model_runner_command_ack",
            "restored": True,
            "envelope_class_identity": True,
        },
        "forbidden_counters": {
            name: 0 for name in preflight.FORBIDDEN_COUNTER_NAMES
        },
        "cuda_initialized_after": False,
    }
    cleanup = {
        "release_rank_order": [3, 2, 1, 0],
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_inert_resources_closed": True,
        "production_exit_call_count": 0,
        "collected_private_objects": {"engine": True},
        "all_private_objects_collected": True,
    }
    memory = {
        "process_before": {"vmrss_kib": 100, "vmhwm_kib": 100},
        "process_ready": {"vmrss_kib": 200, "vmhwm_kib": 300},
        "process_after_cleanup": {"vmrss_kib": 150, "vmhwm_kib": 300},
        "host_before": {"mem_available_kib": 20_000_000},
        "host_ready": {"mem_available_kib": 19_000_000},
    }
    source_hashes = {
        "tools/qwen35_constructed_engine_model_runner_ownership_preflight.py": (
            "1" * 64
        )
    }
    artifact = preflight.build_constructed_runtime_artifact(
        run_tag="constructed-test-20260728-180000",
        smoke=smoke,
        cleanup=cleanup,
        memory=memory,
        source_file_sha256=source_hashes,
        prerequisite_oracle_sha256="2" * 64,
        observed_user="sitian",
        observed_hostname="n232-195-203",
    )

    assert artifact["status"] == "PASS"
    assert artifact["provenance"] == preflight.PROVENANCE
    assert artifact["claim_boundary"] == preflight.CLAIM_BOUNDARY
    assert artifact["rank_payloads_sha256"] == preflight._sha256(
        preflight._canonical(rows)
    )
    assert artifact["cleanup"]["all_private_objects_collected"] is True

    smoke["forbidden_counters"]["inference"] = 1
    _expect_error(
        lambda: preflight.build_constructed_runtime_artifact(
            run_tag="constructed-test-20260728-180001",
            smoke=smoke,
            cleanup=cleanup,
            memory=memory,
            source_file_sha256=source_hashes,
            prerequisite_oracle_sha256="2" * 64,
            observed_user="sitian",
            observed_hostname="n232-195-203",
        ),
        "forbidden counters are non-zero",
    )


def test_finalize_constructed_runtime_artifact_publishes_exact_inventory():
    preflight = _load_preflight()
    artifact = {
        "schema_version": preflight.RESULT_SCHEMA_VERSION,
        "status": "PASS",
        "run_tag": "constructed-test-20260728-180100",
        "prerequisite_oracle_sha256": "2" * 64,
        "source_file_sha256": {"source.py": "1" * 64},
        "source_tree_sha256": preflight._sha256(
            preflight._canonical({"source.py": "1" * 64})
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / artifact["run_tag"]
        result_path, manifest_path = (
            preflight.finalize_constructed_runtime_artifact(
                run_dir=run_dir,
                artifact=artifact,
                remote_target="sitian@10.232.195.203",
                remote_python=(
                    "/data00/home/sitian/sitian-workspace01/tllm/"
                    "env/bin/python"
                ),
            )
        )
        assert {path.name for path in run_dir.iterdir()} == {
            preflight.RESULT_NAME,
            preflight.MANIFEST_NAME,
        }
        manifest = json.loads(manifest_path.read_text())
        assert manifest["result_sha256"] == preflight._sha256(
            result_path.read_bytes()
        )
        assert manifest["source_tree_sha256"] == artifact[
            "source_tree_sha256"
        ]
        _expect_error(
            lambda: preflight.finalize_constructed_runtime_artifact(
                run_dir=run_dir,
                artifact=artifact,
                remote_target="sitian@10.232.195.203",
                remote_python=(
                    "/data00/home/sitian/sitian-workspace01/tllm/"
                    "env/bin/python"
                ),
            ),
            "constructed run directory is not empty",
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "constructed engine/model runner ownership preflight tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
