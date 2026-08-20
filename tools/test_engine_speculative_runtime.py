from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    ROOT / "tinyvllm" / "engine" / "speculative_runtime.py"
)
ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"
GENERIC_SPECULATIVE_PATHS = (
    RUNTIME_PATH,
    ROOT / "tinyvllm" / "engine" / "scheduler.py",
    ROOT / "tinyvllm" / "speculative" / "batch_runtime.py",
    ROOT / "tinyvllm" / "speculative" / "verifier.py",
    ROOT / "tinyvllm" / "engine" / "block_manager.py",
    ROOT / "tinyvllm" / "engine" / "speculative_model_runner.py",
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

adapter_module = sys.modules.get(
    "tinyvllm.speculative.adapter"
)
if adapter_module is None:
    adapter_module = _load_module(
        "tinyvllm.speculative.adapter",
        ROOT / "tinyvllm" / "speculative" / "adapter.py",
    )
selection_module = _load_module(
    "tinyvllm.engine.speculative_selection",
    ROOT
    / "tinyvllm"
    / "engine"
    / "speculative_selection.py",
)
runtime_module = _load_module(
    "tinyvllm.engine.speculative_runtime",
    RUNTIME_PATH,
)
residency_module = _load_module(
    "tinyvllm.engine.speculative_residency",
    ROOT
    / "tinyvllm"
    / "engine"
    / "speculative_residency.py",
)
DraftCapabilities = adapter_module.DraftCapabilities
SpeculativeSelectionConfig = (
    selection_module.SpeculativeSelectionConfig
)
EngineSpeculativeRuntime = (
    runtime_module.EngineSpeculativeRuntime
)
ModelRunnerProposalExecutorDescriptor = getattr(
    runtime_module,
    "ModelRunnerProposalExecutorDescriptor",
    None,
)
validate_engine_speculative_runtime = (
    runtime_module.validate_engine_speculative_runtime
)
build_engine_speculative_selection_config = (
    runtime_module.build_engine_speculative_selection_config
)
KVBlockIdentityRow = residency_module.KVBlockIdentityRow
build_kv_block_identity_rows = (
    residency_module.build_kv_block_identity_rows
)


class SpeculativeKVCommitRollbackError(RuntimeError):
    def __init__(self, commit_error, rollback_error):
        super().__init__(
            "speculative KV commit rollback failed: "
            f"{rollback_error}"
        )
        self.commit_error = commit_error
        self.rollback_error = rollback_error


class SchedulerPostprocessRollbackError(RuntimeError):
    def __init__(self, commit_error, rollback_error):
        super().__init__(
            "scheduler postprocess rollback failed: "
            f"{rollback_error}"
        )
        self.commit_error = commit_error
        self.rollback_error = rollback_error


def _engine_class():
    tree = ast.parse(ENGINE_PATH.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )


def _load_engine_method(
    method_name: str,
    extra_namespace: dict | None = None,
):
    tree = ast.parse(ENGINE_PATH.read_text())
    method = next(
        node
        for node in next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LLMEngine"
        ).body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {
        "EngineSpeculativeRuntime": EngineSpeculativeRuntime,
        "build_engine_speculative_selection_config": (
            build_engine_speculative_selection_config
        ),
        "build_model_runner_proposal_provider": (
            lambda model_runner, runtime, identity_rows_for:
            lambda seqs: ()
        ),
        "build_model_runner_side_state_callbacks": (
            lambda model_runner, dispatch=None: None
        ),
        "apply_prepared_speculative_side_state": (
            lambda prepared: None
        ),
        "rollback_prepared_speculative_side_state": (
            lambda prepared: None
        ),
        "seal_prepared_speculative_side_state": (
            lambda prepared: None
        ),
        "validate_engine_speculative_runtime": (
            validate_engine_speculative_runtime
        ),
        "SpeculativeKVCommitRollbackError": (
            SpeculativeKVCommitRollbackError
        ),
        "SchedulerPostprocessRollbackError": (
            SchedulerPostprocessRollbackError
        ),
    }
    module_body = []
    if method_name == "step":
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "_commit_prepared_speculative_publication"
        )
        namespace.update({
            "build_prepared_proposal_finalize_rows": (
                lambda prepared: ()
            ),
        })
        module_body.append(helper)
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    module_body.append(function)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=module_body, type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def test_generic_speculative_runtime_has_no_model_or_source_dispatch():
    forbidden = (
        "qwen",
        "llama",
        "mtp",
        "learned",
        "checkpoint",
        "source_type ==",
        "source_type in",
    )

    for path in GENERIC_SPECULATIVE_PATHS:
        source = path.read_text().lower()
        for needle in forbidden:
            assert needle not in source, (
                f"{path.relative_to(ROOT)} contains forbidden "
                f"generic dispatch token {needle!r}"
            )


def test_engine_speculative_routing_never_checks_source_type():
    source = ENGINE_PATH.read_text()

    assert "source_type ==" not in source
    assert "source_type in" not in source


class _Adapter:
    def __init__(
        self,
        *,
        supports_batch=True,
        max_proposal_tokens=4,
        requires_target_hidden=False,
        requires_target_logits=False,
        execution_domain="host",
    ):
        self.capabilities = DraftCapabilities(
            source_type="fixture",
            supports_batch=supports_batch,
            requires_target_hidden=requires_target_hidden,
            requires_target_logits=requires_target_logits,
            max_proposal_tokens=max_proposal_tokens,
            execution_domain=execution_domain,
        )

    def propose_batch(self, contexts):
        return ()


class _Lifecycle:
    def register_sequence(self, sequence_id, verified_token_ids):
        pass

    def synchronize_verified_history(
        self,
        sequence_id,
        verified_token_ids,
    ):
        return 0

    def release_sequence(self, sequence_id):
        pass


def _scheduler(*, enabled=True, max_proposal_tokens=4):
    return SimpleNamespace(
        speculative_selection_config=SimpleNamespace(
            enabled=enabled,
            max_proposal_tokens=max_proposal_tokens,
        )
    )


def _model_runner(
    *,
    world_size=1,
    kv_offload_mvp0=False,
):
    return SimpleNamespace(
        call=lambda *args, **kwargs: None,
        world_size=world_size,
        config=SimpleNamespace(
            kv_offload_mvp0=kv_offload_mvp0,
        ),
    )


def test_engine_speculative_runtime_contract_exists():
    assert RUNTIME_PATH.exists()
    tree = ast.parse(RUNTIME_PATH.read_text())
    names = {
        node.name
        for node in tree.body
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef),
        )
    }
    assert "DraftLifecycle" in names
    assert "EngineSpeculativeRuntime" in names
    assert "ModelRunnerProposalExecutorDescriptor" in names
    assert "validate_engine_speculative_runtime" in names


def test_llm_engine_exposes_runtime_installation():
    method_names = {
        node.name
        for node in _engine_class().body
        if isinstance(node, ast.FunctionDef)
    }
    assert "install_speculative_runtime" in method_names


def test_kv_offload_identity_rows_follow_allocator_generations():
    method = _load_engine_method(
        "_kv_offload_identity_rows",
        {
            "build_kv_block_identity_rows": (
                build_kv_block_identity_rows
            ),
        },
    )
    calls = []
    block_manager = SimpleNamespace(
        block_identities=(
            lambda block_ids:
            calls.append(block_ids)
            or tuple(
                (block_id, block_id + 10)
                for block_id in block_ids
            )
        )
    )
    seqs = (
        SimpleNamespace(seq_id=7, block_table=[1, 3]),
        SimpleNamespace(seq_id=9, block_table=[5]),
    )
    enabled_engine = SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(kv_offload_mvp0=True)
        ),
        scheduler=SimpleNamespace(
            block_manager=block_manager
        ),
    )
    disabled_engine = SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(kv_offload_mvp0=False)
        ),
        scheduler=SimpleNamespace(
            block_manager=block_manager
        ),
    )

    assert method(disabled_engine, seqs) == ()
    assert calls == []
    assert method(enabled_engine, seqs) == (
        KVBlockIdentityRow(7, ((1, 11), (3, 13))),
        KVBlockIdentityRow(9, ((5, 15),)),
    )
    assert calls == [(1, 3), (5,)]


def test_residency_phase_acknowledgements_are_exact_and_rank_complete():
    method = _load_engine_method(
        "_call_speculative_residency_phase"
    )
    expected_rows = (
        {
            "ticket_id": 41,
            "participant_id": 0,
            "operation": "prepare",
            "status": "prepared",
            "sequence_ids": (7,),
            "committed_block_identities": (),
            "rejected_block_identities": (),
            "detail": "",
        },
        {
            "ticket_id": 41,
            "participant_id": 1,
            "operation": "prepare",
            "status": "prepared",
            "sequence_ids": (7,),
            "committed_block_identities": (),
            "rejected_block_identities": (),
            "detail": "",
        },
    )
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=2),
        call_model_runner_acknowledged=(
            lambda method_name, *args, timeout_s: (
                expected_rows[0],
                (
                    SimpleNamespace(
                        rank=1,
                        result=expected_rows[1],
                    ),
                ),
            )
        ),
        model_runner_ack_collector=SimpleNamespace(
            poison=lambda reason: (_ for _ in ()).throw(
                AssertionError(reason)
            )
        ),
    )

    rows = method(
        engine,
        "prepare_speculative_residency_batch",
        41,
        (("row",),),
        expected_operation="prepare",
        expected_status="prepared",
        expected_sequence_ids=(7,),
        expected_committed_block_identities=(),
        expected_rejected_block_identities=(),
        timeout_s=60.0,
    )

    assert rows == expected_rows


def test_side_state_phase_waits_for_all_rank_acknowledgements():
    method = _load_engine_method(
        "_call_speculative_side_state_phase"
    )
    calls = []
    local = {
        "operation": "seal",
        "status": "sealed",
        "transaction_id": "tx-1",
        "sequence_ids": [7],
    }
    worker_rows = (
        SimpleNamespace(
            rank=1,
            result={
                "operation": "seal",
                "status": "sealed",
                "transaction_id": "tx-1",
                "sequence_ids": [7],
            },
        ),
    )
    engine = SimpleNamespace(
        call_model_runner_acknowledged=(
            lambda method_name, *args, timeout_s: (
                calls.append(
                    (method_name, args, timeout_s)
                )
                or (local, worker_rows)
            )
        ),
    )

    assert method(
        engine,
        "seal_speculative_side_state_batch",
    ) == local
    assert calls == [(
        "seal_speculative_side_state_batch",
        (),
        60.0,
    )]


def test_step_wires_side_state_callbacks_to_acknowledged_phase():
    step = next(
        node
        for node in _engine_class().body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )
    builder_call = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "build_model_runner_side_state_callbacks"
    )
    dispatch_keyword = next(
        keyword
        for keyword in builder_call.keywords
        if keyword.arg == "dispatch"
    )

    assert isinstance(dispatch_keyword.value, ast.Lambda)
    helper_calls = [
        node
        for node in ast.walk(dispatch_keyword.value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        == "_call_speculative_side_state_phase"
    ]
    assert len(helper_calls) == 1


def test_malformed_residency_acknowledgement_poisons_collector():
    method = _load_engine_method(
        "_call_speculative_residency_phase"
    )
    poison_reasons = []
    malformed = {
        "ticket_id": 41,
        "participant_id": 0,
        "operation": "prepare",
        "status": "prepared",
        "sequence_ids": (7,),
        "committed_block_identities": (),
        "rejected_block_identities": (),
        "detail": "",
        "extra": True,
    }
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=1),
        call_model_runner_acknowledged=(
            lambda *args, **kwargs: (malformed, ())
        ),
        model_runner_ack_collector=SimpleNamespace(
            poison=poison_reasons.append
        ),
    )

    with pytest.raises(RuntimeError, match="acknowledgement"):
        method(
            engine,
            "prepare_speculative_residency_batch",
            41,
            (),
            expected_operation="prepare",
            expected_status="prepared",
            expected_sequence_ids=(7,),
            expected_committed_block_identities=(),
            expected_rejected_block_identities=(),
            timeout_s=60.0,
        )

    assert len(poison_reasons) == 1
    assert "invalid" in poison_reasons[0]


def test_kv_offload_summaries_collect_rank_order():
    method = _load_engine_method("kv_offload_summaries")
    rank_zero = {"h2d_copies": 3}
    rank_one = {"h2d_copies": 4}
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=2),
        call_model_runner_acknowledged=(
            lambda method_name, timeout_s: (
                rank_zero,
                (
                    SimpleNamespace(
                        rank=1,
                        result=rank_one,
                    ),
                ),
            )
        ),
    )

    assert method(engine, timeout_s=60.0) == (
        rank_zero,
        rank_one,
    )


def test_reset_peak_memory_stats_returns_rank_ordered_rows():
    method = _load_engine_method("reset_peak_memory_stats")
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=2),
        call_model_runner_acknowledged=(
            lambda method_name, timeout_s: (
                {
                    "cuda_allocated_bytes": 10,
                    "cuda_peak_allocated_bytes": 10,
                },
                (
                    SimpleNamespace(
                        rank=1,
                        result={
                            "cuda_allocated_bytes": 20,
                            "cuda_peak_allocated_bytes": 20,
                        },
                    ),
                ),
            )
        ),
    )

    rows = method(engine, timeout_s=3.0)

    assert rows == (
        {
            "cuda_allocated_bytes": 10,
            "cuda_peak_allocated_bytes": 10,
            "rank": 0,
        },
        {
            "cuda_allocated_bytes": 20,
            "cuda_peak_allocated_bytes": 20,
            "rank": 1,
        },
    )


@pytest.mark.parametrize(
    "local_result,worker_acks,match",
    (
        (
            [],
            (),
            "rank mismatch",
        ),
        (
            {"rank": 1},
            (),
            "rank mismatch",
        ),
        (
            {},
            (
                SimpleNamespace(rank=0, result={}),
            ),
            "rank mismatch",
        ),
        (
            {},
            (),
            "inventory mismatch",
        ),
    ),
)
def test_reset_peak_memory_stats_rejects_invalid_rank_inventory(
    local_result,
    worker_acks,
    match,
):
    method = _load_engine_method("reset_peak_memory_stats")
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=2),
        call_model_runner_acknowledged=(
            lambda method_name, timeout_s: (
                local_result,
                worker_acks,
            )
        ),
    )

    with pytest.raises(ValueError, match=match):
        method(engine, timeout_s=3.0)


def test_runtime_validation_accepts_generic_batch_adapter():
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(),
        lifecycle=_Lifecycle(),
    )

    assert validate_engine_speculative_runtime(
        runtime,
        scheduler=_scheduler(),
        model_runner=_model_runner(),
    ) is runtime


def _model_runner_descriptor(**capability_overrides):
    values = {
        "source_type": "fixture",
        "supports_batch": True,
        "requires_target_hidden": True,
        "requires_target_logits": False,
        "max_proposal_tokens": 4,
        "execution_domain": "model_runner",
    }
    values.update(capability_overrides)
    return ModelRunnerProposalExecutorDescriptor(
        executor_id="fixture-executor",
        capabilities=DraftCapabilities(**values),
    )


def test_runtime_capabilities_follow_configured_source():
    host = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(),
    )
    device = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(),
    )

    assert host.capabilities.execution_domain == "host"
    assert (
        device.capabilities.execution_domain
        == "model_runner"
    )


@pytest.mark.parametrize(
    "runtime",
    (
        EngineSpeculativeRuntime(
            draft_adapter=None,
        ),
        EngineSpeculativeRuntime(
            draft_adapter=_Adapter(),
            model_runner_executor=SimpleNamespace(),
        ),
    ),
)
def test_runtime_requires_exactly_one_proposal_source(runtime):
    with pytest.raises(ValueError, match="exactly one"):
        build_engine_speculative_selection_config(
            runtime,
            model_runner=_model_runner(),
        )


def test_runtime_preparation_accepts_model_runner_descriptor():
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(),
    )

    config = build_engine_speculative_selection_config(
        runtime,
        model_runner=_model_runner(),
    )

    assert config == SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )


def test_model_runner_runtime_accepts_tp4_without_offload():
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(),
    )

    config = build_engine_speculative_selection_config(
        runtime,
        model_runner=_model_runner(world_size=4),
    )

    assert config == SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )


def test_model_runner_runtime_tp4_rejects_target_logits():
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(
            requires_target_logits=True,
        ),
    )

    with pytest.raises(ValueError, match="target logits"):
        build_engine_speculative_selection_config(
            runtime,
            model_runner=_model_runner(world_size=4),
        )


@pytest.mark.parametrize("world_size", (2, 3, 8))
def test_model_runner_runtime_rejects_unsupported_world_size(
    world_size,
):
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(),
    )

    with pytest.raises(ValueError, match="TP1 or TP4"):
        build_engine_speculative_selection_config(
            runtime,
            model_runner=_model_runner(world_size=world_size),
        )


def test_model_runner_runtime_accepts_target_kv_offload():
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=_model_runner_descriptor(),
    )

    config = build_engine_speculative_selection_config(
        runtime,
        model_runner=_model_runner(
            kv_offload_mvp0=True,
        ),
    )

    assert config == SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )


def test_runtime_preparation_derives_exact_selection_config():
    builder = getattr(
        runtime_module,
        "build_engine_speculative_selection_config",
        None,
    )
    assert builder is not None
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
        lifecycle=_Lifecycle(),
    )

    config = builder(
        runtime,
        model_runner=_model_runner(),
    )

    assert config == SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )


def test_runtime_preparation_rejects_limit_one():
    builder = getattr(
        runtime_module,
        "build_engine_speculative_selection_config",
        None,
    )
    assert builder is not None
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=1),
    )

    with pytest.raises(
        ValueError,
        match="max_proposal_tokens >= 2",
    ):
        builder(
            runtime,
            model_runner=_model_runner(),
        )


@pytest.mark.parametrize(
    ("runtime", "scheduler", "message"),
    (
        (
            EngineSpeculativeRuntime(
                draft_adapter=_Adapter(
                    supports_batch=False,
                )
            ),
            _scheduler(),
            "batch",
        ),
        (
            EngineSpeculativeRuntime(
                draft_adapter=_Adapter(
                    max_proposal_tokens=3,
                )
            ),
            _scheduler(max_proposal_tokens=4),
            "proposal limit",
        ),
        (
            EngineSpeculativeRuntime(
                draft_adapter=_Adapter()
            ),
            _scheduler(enabled=False),
            "selection",
        ),
    ),
)
def test_runtime_validation_rejects_incompatible_installation(
    runtime,
    scheduler,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_engine_speculative_runtime(
            runtime,
            scheduler=scheduler,
            model_runner=_model_runner(),
        )


def test_runtime_validation_rejects_incomplete_lifecycle():
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(),
        lifecycle=SimpleNamespace(
            register_sequence=lambda *args: None,
        ),
    )

    with pytest.raises(ValueError, match="lifecycle"):
        validate_engine_speculative_runtime(
            runtime,
            scheduler=_scheduler(),
            model_runner=_model_runner(),
        )


def test_runtime_installation_is_same_object_idempotent():
    install = _load_engine_method(
        "install_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(),
    )
    engine = SimpleNamespace(
        scheduler=_scheduler(),
        model_runner=_model_runner(),
        speculative_runtime=None,
        speculative_runtime_poisoned=True,
        speculative_runtime_poison_reason="stale",
    )

    install(engine, runtime)
    install(engine, runtime)

    assert engine.speculative_runtime is runtime
    assert engine.speculative_runtime_poisoned is False
    assert engine.speculative_runtime_poison_reason is None
    with pytest.raises(RuntimeError, match="already installed"):
        install(
            engine,
            EngineSpeculativeRuntime(
                draft_adapter=_Adapter(),
            ),
        )


class _ActivationScheduler:
    def __init__(self):
        self.speculative_selection_config = (
            SpeculativeSelectionConfig(
                enabled=False,
                max_proposal_tokens=0,
            )
        )
        self._speculative_selection_installed = False
        self.fail_install = False

    def install_speculative_selection(self, config):
        if self.fail_install:
            raise RuntimeError(
                "injected selection publication failure"
            )
        if not self._speculative_selection_installed:
            self.speculative_selection_config = config
            self._speculative_selection_installed = True
            return
        if self.speculative_selection_config == config:
            return
        raise RuntimeError(
            "speculative selection config is already installed"
        )


def _activation_engine():
    engine = SimpleNamespace(
        scheduler=_ActivationScheduler(),
        model_runner=_model_runner(),
        speculative_runtime=None,
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
    )

    def install_runtime(runtime):
        if engine.speculative_runtime is runtime:
            return
        if engine.speculative_runtime is not None:
            raise RuntimeError(
                "speculative runtime is already installed"
            )
        validate_engine_speculative_runtime(
            runtime,
            scheduler=engine.scheduler,
            model_runner=engine.model_runner,
        )
        engine.speculative_runtime = runtime
        engine.speculative_runtime_poisoned = False
        engine.speculative_runtime_poison_reason = None

    engine.install_speculative_runtime = install_runtime
    return engine


def test_atomic_activation_publishes_matching_runtime_and_selection():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()

    activate(engine, runtime)

    assert engine.speculative_runtime is runtime
    assert engine.scheduler.speculative_selection_config == (
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )
    assert engine.scheduler._speculative_selection_installed


def test_atomic_activation_is_same_object_idempotent():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()

    activate(engine, runtime)
    activate(engine, runtime)

    assert engine.speculative_runtime is runtime
    assert engine.scheduler.speculative_selection_config == (
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )


def test_atomic_activation_rejects_conflicting_runtime_before_mutation():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    installed = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    candidate = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    activate(engine, installed)
    selection_before = (
        engine.scheduler.speculative_selection_config
    )

    with pytest.raises(
        RuntimeError,
        match="already installed",
    ):
        activate(engine, candidate)

    assert engine.speculative_runtime is installed
    assert (
        engine.scheduler.speculative_selection_config
        == selection_before
    )


def test_atomic_activation_rejects_conflicting_selection_before_mutation():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    engine.scheduler.speculative_selection_config = (
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=3,
        )
    )
    engine.scheduler._speculative_selection_installed = True

    with pytest.raises(
        RuntimeError,
        match="selection config is already installed",
    ):
        activate(engine, runtime)

    assert engine.speculative_runtime is None
    assert (
        engine.scheduler.speculative_selection_config
        == SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=3,
        )
    )


def test_atomic_activation_rolls_back_scheduler_on_runtime_failure():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    original_selection = (
        engine.scheduler.speculative_selection_config
    )
    engine.speculative_runtime_poisoned = True
    engine.speculative_runtime_poison_reason = "existing"

    def fail_runtime_install(candidate):
        del candidate
        raise RuntimeError(
            "injected runtime publication failure"
        )

    engine.install_speculative_runtime = fail_runtime_install

    with pytest.raises(
        RuntimeError,
        match="runtime publication failure",
    ):
        activate(engine, runtime)

    assert engine.speculative_runtime is None
    assert (
        engine.scheduler.speculative_selection_config
        == original_selection
    )
    assert not engine.scheduler._speculative_selection_installed
    assert engine.speculative_runtime_poisoned is True
    assert (
        engine.speculative_runtime_poison_reason
        == "existing"
    )


def test_atomic_activation_rolls_back_on_scheduler_failure():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    runtime = EngineSpeculativeRuntime(
        draft_adapter=_Adapter(max_proposal_tokens=4),
    )
    engine = _activation_engine()
    original_selection = (
        engine.scheduler.speculative_selection_config
    )
    engine.scheduler.fail_install = True

    with pytest.raises(
        RuntimeError,
        match="selection publication failure",
    ):
        activate(engine, runtime)

    assert engine.speculative_runtime is None
    assert (
        engine.scheduler.speculative_selection_config
        == original_selection
    )
    assert not engine.scheduler._speculative_selection_installed


def test_atomic_activation_invalid_runtime_leaves_state_unchanged():
    activate = _load_engine_method(
        "activate_speculative_runtime"
    )
    engine = _activation_engine()
    original_selection = (
        engine.scheduler.speculative_selection_config
    )

    with pytest.raises(ValueError, match="batch"):
        activate(
            engine,
            EngineSpeculativeRuntime(
                draft_adapter=_Adapter(
                    supports_batch=False,
                ),
            ),
        )

    assert engine.speculative_runtime is None
    assert (
        engine.scheduler.speculative_selection_config
        == original_selection
    )
    assert not engine.scheduler._speculative_selection_installed


def test_add_request_registers_lifecycle_before_scheduler_admission():
    events = []

    class Sequence:
        next_id = 40

        def __init__(self, token_ids, sampling_params):
            del sampling_params
            self.seq_id = self.next_id
            type(self).next_id += 1
            self.token_ids = list(token_ids)

    class Lifecycle:
        def register_sequence(
            self,
            sequence_id,
            verified_token_ids,
        ):
            events.append((
                "register",
                sequence_id,
                verified_token_ids,
            ))

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    def scheduler_add(sequence):
        events.append(("add", sequence.seq_id))

    add_request = _load_engine_method(
        "add_request",
        {
            "Sequence": Sequence,
            "_try_qwen35_hybrid_prefix_restore": (
                lambda engine, sequence: None
            ),
        },
    )
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(
            encode=lambda prompt: [1, 2]
        ),
        speculative_runtime=SimpleNamespace(
            lifecycle=Lifecycle()
        ),
        scheduler=SimpleNamespace(add=scheduler_add),
    )

    add_request(engine, [1, 2, 3], object())

    assert events == [
        ("register", 40, (1, 2, 3)),
        ("add", 40),
    ]


def test_add_request_releases_lifecycle_when_admission_fails():
    events = []

    class Sequence:
        def __init__(self, token_ids, sampling_params):
            del sampling_params
            self.seq_id = 50
            self.token_ids = list(token_ids)

    class Lifecycle:
        def register_sequence(
            self,
            sequence_id,
            verified_token_ids,
        ):
            events.append(("register", sequence_id))

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    def fail_add(sequence):
        events.append(("add", sequence.seq_id))
        raise RuntimeError("injected admission failure")

    add_request = _load_engine_method(
        "add_request",
        {
            "Sequence": Sequence,
            "_try_qwen35_hybrid_prefix_restore": (
                lambda engine, sequence: None
            ),
        },
    )
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(
            encode=lambda prompt: [1, 2]
        ),
        speculative_runtime=SimpleNamespace(
            lifecycle=Lifecycle()
        ),
        scheduler=SimpleNamespace(add=fail_add),
    )

    with pytest.raises(
        RuntimeError,
        match="injected admission failure",
    ):
        add_request(engine, [1, 2, 3], object())

    assert events == [
        ("register", 50),
        ("add", 50),
        ("release", 50),
    ]


@dataclass(frozen=True)
class _ScheduledOutputRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    speculative: bool
    accepted_draft_tokens: tuple[int, ...] = ()


class _Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_step_releases_lifecycle_after_ordinary_only_finish():
    events = []

    class Sequence:
        seq_id = 7
        num_prompt_tokens = 2
        token_ids = [1, 2]
        block_table = [7]
        prefill_chunk_start = 0
        prefill_chunk_end = 0
        prefill_chunk_final = False
        step_is_decode = True
        step_do_sample = True
        status = "running"

        @property
        def completion_token_ids(self):
            return self.token_ids[self.num_prompt_tokens:]

        @property
        def is_finished(self):
            return self.status == "finished"

    sequence = Sequence()

    class Lifecycle:
        def synchronize_verified_history(
            self,
            sequence_id,
            token_ids,
        ):
            events.append(("synchronize", sequence_id, token_ids))

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    class Scheduler:
        last_policy_branch = "decode"
        last_speculative_selection = object()
        schedule_generation = 3

        def observation_snapshot(self):
            return {"running_seq_ids": [7]}

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 10
            return [sequence], False, True

        def drain_hybrid_state_release_events(self):
            return ()

        def postprocess(
            self,
            seqs,
            token_ids,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == [sequence]
            assert token_ids == [11]
            assert (is_prefill, do_sample, batch_kind) == (
                False,
                True,
                None,
            )
            assert (decision_now_ns, step_end_ns) == (10, 20)
            sequence.token_ids.append(11)
            sequence.status = "finished"

        def last_slo_observation(self):
            return {
                "decision_now_ns": 10,
                "step_end_ns": 20,
                "actual_step_duration_ns": 10,
            }

    partition = SimpleNamespace(
        schedule_generation=3,
        selected_sequence_ids=(),
        suppressed_sequence_ids=(7,),
        selected_sequences=(),
        suppressed_sequences=(sequence,),
    )
    step = _load_engine_method(
        "step",
        {
            "build_engine_speculative_partition": (
                lambda *args, **kwargs: partition
            ),
        },
    )
    engine = SimpleNamespace(
        _clock_ns=_Clock(10, 20),
        scheduler=Scheduler(),
        model_runner=SimpleNamespace(
            call=lambda *args, **kwargs: [11],
            memory_snapshot=lambda: {"bytes": 1},
        ),
        speculative_runtime=SimpleNamespace(
            lifecycle=Lifecycle(),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )

    assert step(engine) == ([(7, [11])], -1)
    assert events == [
        ("synchronize", 7, (1, 2, 11)),
        ("release", 7),
    ]


def test_step_releases_model_runner_proposal_after_ordinary_only_finish():
    events = []

    class Sequence:
        seq_id = 7
        sequence_epoch = 5
        num_prompt_tokens = 2
        token_ids = [1, 2]
        block_table = [7]
        prefill_chunk_start = 0
        prefill_chunk_end = 0
        prefill_chunk_final = False
        step_is_decode = True
        step_do_sample = True
        status = "running"

        @property
        def completion_token_ids(self):
            return self.token_ids[self.num_prompt_tokens:]

        @property
        def is_finished(self):
            return self.status == "finished"

    sequence = Sequence()

    class Lifecycle:
        def synchronize_verified_history(
            self,
            sequence_id,
            token_ids,
        ):
            events.append(("synchronize", sequence_id, token_ids))

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    class Scheduler:
        last_policy_branch = "decode"
        last_speculative_selection = object()
        schedule_generation = 3

        def observation_snapshot(self):
            return {"running_seq_ids": [7]}

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 10
            return [sequence], False, True

        def drain_hybrid_state_release_events(self):
            return ()

        def postprocess(
            self,
            seqs,
            token_ids,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == [sequence]
            assert token_ids == [11]
            assert (is_prefill, do_sample, batch_kind) == (
                False,
                True,
                None,
            )
            assert (decision_now_ns, step_end_ns) == (10, 20)
            sequence.token_ids.append(11)
            sequence.status = "finished"

        def last_slo_observation(self):
            return {
                "decision_now_ns": 10,
                "step_end_ns": 20,
                "actual_step_duration_ns": 10,
            }

    def release_proposal_sequence(
        model_runner,
        descriptor,
        sequence_id,
        sequence_epoch,
        *,
        dispatch,
    ):
        assert model_runner is engine.model_runner
        dispatch(
            "release_speculative_proposal_sequence",
            descriptor.executor_id,
            sequence_id,
            sequence_epoch,
        )

    partition = SimpleNamespace(
        schedule_generation=3,
        selected_sequence_ids=(),
        suppressed_sequence_ids=(7,),
        selected_sequences=(),
        suppressed_sequences=(sequence,),
    )
    step = _load_engine_method(
        "step",
        {
            "build_engine_speculative_partition": (
                lambda *args, **kwargs: partition
            ),
            "release_model_runner_proposal_sequence": (
                release_proposal_sequence
            ),
            "_call_speculative_proposal_lifecycle": (
                lambda engine, method_name, *args: events.append(
                    ("proposal_dispatch", method_name, args)
                )
            ),
        },
    )
    engine = SimpleNamespace(
        _clock_ns=_Clock(10, 20),
        scheduler=Scheduler(),
        model_runner=SimpleNamespace(
            call=lambda *args, **kwargs: [11],
            memory_snapshot=lambda: {"bytes": 1},
        ),
        speculative_runtime=EngineSpeculativeRuntime(
            model_runner_executor=_model_runner_descriptor(
                requires_target_hidden=False,
                requires_proposal_lifecycle=True,
            ),
            lifecycle=Lifecycle(),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )

    assert step(engine) == ([(7, [11])], -1)
    assert events == [
        ("synchronize", 7, (1, 2, 11)),
        ("release", 7),
        (
            "proposal_dispatch",
            "release_speculative_proposal_sequence",
            ("fixture-executor", 7, 5),
        ),
    ]


def test_step_executes_selected_runtime_and_commits_multi_token_row():
    events = []

    class Sequence:
        def __init__(self):
            self.seq_id = 7
            self.token_ids = [1, 2]
            self.num_prompt_tokens = 2
            self.prefill_chunk_start = 0
            self.prefill_chunk_end = 0
            self.prefill_chunk_final = False
            self.step_is_decode = False
            self.step_do_sample = True
            self.status = "running"

        @property
        def completion_token_ids(self):
            return self.token_ids[self.num_prompt_tokens:]

        @property
        def is_finished(self):
            return False

    sequence = Sequence()
    prepared_runtime = SimpleNamespace(
        sequences=(
            SimpleNamespace(
                sequence_id=7,
                transaction=None,
                accepted_tokens=(11, 12),
                proposal=SimpleNamespace(
                    token_ids=(11, 12, 14),
                ),
            ),
        ),
        timing_ms={"draft_proposal_ms": 1.0},
        first_target_callback_count=1,
        tail_callback_count=1,
        side_state_callbacks=None,
        side_state_state=None,
        state="prepared",
    )
    commit_row = SimpleNamespace(
        sequence_id=7,
        output_tokens=(11, 12, 13),
        accepted_draft_tokens=(11, 12),
    )
    prepared_scheduler = SimpleNamespace(
        state="prepared",
        snapshot=SimpleNamespace(
            extend_speculative_kv_plans=(
                lambda scheduler, plans: events.append(
                    ("scheduler_journal_extend", plans)
                )
            )
        ),
    )

    class Scheduler:
        last_policy_branch = "decode"
        last_speculative_selection = object()
        schedule_generation = 3
        eos = -1

        def __init__(self):
            self.block_manager = SimpleNamespace(
                prepare_speculative_kv_commit=lambda *args: None,
                commit_speculative_kv_commit_batch=lambda plans: (
                    events.append(("kv_commit", plans))
                ),
            )

        def observation_snapshot(self):
            return {"running_seq_ids": [7]}

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 10
            return [sequence], False, True

        def drain_hybrid_state_release_events(self):
            raise AssertionError(
                "selected-only path must flush releases explicitly"
            )

        def prepare_postprocess(
            self,
            seqs,
            rows,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == (sequence,)
            assert rows == (
                _ScheduledOutputRow(
                    sequence_id=7,
                    output_tokens=(11, 12, 13),
                    speculative=True,
                    accepted_draft_tokens=(11, 12),
                ),
            )
            assert (is_prefill, do_sample, batch_kind) == (
                False,
                True,
                None,
            )
            assert (decision_now_ns, step_end_ns) == (10, 20)
            events.append(("scheduler_prepare", rows))
            return prepared_scheduler

        def commit_prepared_postprocess(self, prepared):
            assert prepared is prepared_scheduler
            sequence.token_ids.extend((11, 12, 13))
            prepared.state = "committed"
            events.append(("scheduler_commit",))

        def last_slo_observation(self):
            return {
                "decision_now_ns": 10,
                "step_end_ns": 20,
                "actual_step_duration_ns": 10,
            }

    partition = SimpleNamespace(
        schedule_generation=3,
        selected_sequence_ids=(7,),
        suppressed_sequence_ids=(),
        selected_sequences=(sequence,),
        suppressed_sequences=(),
    )

    proposal_provider = lambda seqs: ()

    def build_proposal_provider(
        model_runner,
        runtime,
        identity_rows_for,
    ):
        assert model_runner is engine.model_runner
        assert runtime is engine.speculative_runtime
        assert callable(identity_rows_for)
        events.append(("provider_build",))
        return proposal_provider

    def prepare_runtime(**kwargs):
        assert kwargs["seqs"] == (sequence,)
        assert (
            kwargs["run_first_targets_and_proposals"]
            is proposal_provider
        )
        assert "draft_adapter" not in kwargs
        assert "run_first_targets" not in kwargs
        events.append(("runtime_prepare",))
        return prepared_runtime

    step = _load_engine_method(
        "step",
        {
            "build_engine_speculative_partition": (
                lambda *args, **kwargs: partition
            ),
            "build_model_runner_proposal_provider": (
                build_proposal_provider
            ),
            "build_model_runner_side_state_callbacks": (
                lambda model_runner, dispatch=None: None
            ),
            "apply_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "rollback_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "seal_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "prepare_native_speculative_batch": prepare_runtime,
            "rollback_prepared_native_speculative_batch": (
                lambda **kwargs: events.append(
                    ("runtime_rollback",)
                )
            ),
            "build_engine_prepared_speculative_commit_rows": (
                lambda *args, **kwargs: (commit_row,)
            ),
            "run_model_runner_tail_batch": (
                lambda *args, **kwargs: ()
            ),
            "ScheduledOutputRow": _ScheduledOutputRow,
        },
    )
    engine = SimpleNamespace(
        _clock_ns=_Clock(10, 20),
        scheduler=Scheduler(),
        model_runner=SimpleNamespace(
            memory_snapshot=lambda: {"bytes": 1}
        ),
        speculative_runtime=EngineSpeculativeRuntime(
            draft_adapter=_Adapter(),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
        flush_pending_hybrid_state_releases=(
            lambda **kwargs: events.append(
                ("flush_releases",)
            )
        ),
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )

    result = step(engine)

    assert result == ([], -1)
    assert sequence.completion_token_ids == [11, 12, 13]
    assert prepared_runtime.state == "committed"
    assert events == [
        ("flush_releases",),
        ("provider_build",),
        ("runtime_prepare",),
        ("scheduler_prepare", (
            _ScheduledOutputRow(
                sequence_id=7,
                output_tokens=(11, 12, 13),
                speculative=True,
                accepted_draft_tokens=(11, 12),
            ),
        )),
        ("scheduler_commit",),
    ]
    assert engine.last_step_observation[
        "new_completion_tokens_by_seq"
    ] == {7: [11, 12, 13]}
    assert engine.last_step_observation[
        "speculative_proposal_token_counts"
    ] == {7: 3}
    assert engine.last_step_observation[
        "speculative_proposal_token_ids_by_seq"
    ] == {7: [11, 12, 14]}
    assert engine.last_step_observation[
        "speculative_accepted_draft_token_ids_by_seq"
    ] == {7: [11, 12]}
    assert engine.last_step_observation[
        "speculative_proposal_row_count"
    ] == 1
    assert engine.last_step_observation[
        "speculative_first_target_callback_count"
    ] == 1


def test_step_merges_selected_and_suppressed_rows_in_schedule_order():
    events = []

    class Lifecycle:
        def register_sequence(self, sequence_id, token_ids):
            raise AssertionError("step must not register lifecycle rows")

        def synchronize_verified_history(
            self,
            sequence_id,
            token_ids,
        ):
            events.append(("synchronize", sequence_id, token_ids))
            return len(token_ids)

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    class Sequence:
        def __init__(self, sequence_id):
            self.seq_id = sequence_id
            self.token_ids = [1, 2]
            self.block_table = [sequence_id]
            self.num_prompt_tokens = 2
            self.prefill_chunk_start = 0
            self.prefill_chunk_end = 0
            self.prefill_chunk_final = False
            self.step_is_decode = True
            self.step_do_sample = True
            self.status = "running"

        @property
        def completion_token_ids(self):
            return self.token_ids[self.num_prompt_tokens:]

        @property
        def is_finished(self):
            return False

    selected = Sequence(7)
    suppressed = Sequence(8)
    prepared_runtime = SimpleNamespace(
        sequences=(
            SimpleNamespace(
                sequence_id=7,
                transaction=None,
                accepted_tokens=(11, 12),
                proposal=SimpleNamespace(
                    token_ids=(11, 12, 14),
                ),
            ),
        ),
        timing_ms={"draft_proposal_ms": 1.0},
        first_target_callback_count=1,
        tail_callback_count=1,
        side_state_callbacks=None,
        side_state_state="disabled",
        state="prepared",
    )
    commit_row = SimpleNamespace(
        sequence_id=7,
        output_tokens=(11, 12, 13),
        accepted_draft_tokens=(11, 12),
    )
    prepared_scheduler = SimpleNamespace(
        state="prepared",
        snapshot=SimpleNamespace(
            extend_speculative_kv_plans=(
                lambda scheduler, plans: events.append(
                    ("scheduler_journal_extend", plans)
                )
            )
        ),
    )

    class ModelRunner:
        config = SimpleNamespace(kv_offload_mvp0=True)

        def call(self, method, *args):
            assert method == "run"
            assert args == (
                (suppressed,),
                False,
                True,
                "mixed",
                ("lease",),
                (
                    KVBlockIdentityRow(
                        8,
                        ((8, 108),),
                    ),
                ),
            )
            events.append(("ordinary_run", args))
            return (21,)

        def memory_snapshot(self):
            return {"bytes": 1}

    class Scheduler:
        last_policy_branch = "mixed"
        last_speculative_selection = object()
        schedule_generation = 4
        eos = -1

        def __init__(self):
            self.block_manager = SimpleNamespace(
                prepare_speculative_kv_commit=lambda *args: None,
                commit_speculative_kv_commit_batch=lambda plans: None,
                block_identities=(
                    lambda block_ids:
                    tuple(
                        (block_id, block_id + 100)
                        for block_id in block_ids
                    )
                ),
            )

        def observation_snapshot(self):
            return {"running_seq_ids": [7, 8]}

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 10
            return [selected, suppressed], False, True, "mixed"

        def drain_hybrid_state_release_events(self):
            events.append(("drain_releases",))
            return ("lease",)

        def restore_hybrid_state_release_events(self, releases):
            events.append(("restore_releases", releases))

        def prepare_postprocess(
            self,
            seqs,
            rows,
            is_prefill,
            do_sample,
            batch_kind,
            *,
            decision_now_ns,
            step_end_ns,
        ):
            assert seqs == (selected, suppressed)
            assert rows == (
                _ScheduledOutputRow(
                    sequence_id=7,
                    output_tokens=(11, 12, 13),
                    speculative=True,
                    accepted_draft_tokens=(11, 12),
                ),
                _ScheduledOutputRow(
                    sequence_id=8,
                    output_tokens=(21,),
                    speculative=False,
                ),
            )
            assert (is_prefill, do_sample, batch_kind) == (
                False,
                True,
                "mixed",
            )
            assert (decision_now_ns, step_end_ns) == (10, 20)
            events.append(("scheduler_prepare", rows))
            return prepared_scheduler

        def commit_prepared_postprocess(self, prepared):
            assert prepared is prepared_scheduler
            selected.token_ids.extend((11, 12, 13))
            suppressed.token_ids.append(21)
            selected.step_is_decode = False
            suppressed.step_is_decode = False
            prepared.state = "committed"
            events.append(("scheduler_commit",))

        def last_slo_observation(self):
            return {
                "decision_now_ns": 10,
                "step_end_ns": 20,
                "actual_step_duration_ns": 10,
            }

    partition = SimpleNamespace(
        schedule_generation=4,
        selected_sequence_ids=(7,),
        suppressed_sequence_ids=(8,),
        selected_sequences=(selected,),
        suppressed_sequences=(suppressed,),
    )

    def run_proposals(
        model_runner,
        seqs,
        identity_rows_for,
    ):
        identity_rows = identity_rows_for(seqs)
        assert model_runner is engine.model_runner
        assert seqs == (selected,)
        assert identity_rows == (
            KVBlockIdentityRow(
                7,
                ((7, 107),),
            ),
        )
        events.append(("first_targets", identity_rows))
        return ()

    def build_proposal_provider(
        model_runner,
        runtime,
        identity_rows_for,
    ):
        assert model_runner is engine.model_runner
        assert runtime is engine.speculative_runtime
        return lambda seqs: run_proposals(
            model_runner,
            seqs,
            identity_rows_for,
        )

    def prepare_runtime(**kwargs):
        assert "draft_adapter" not in kwargs
        assert "run_first_targets" not in kwargs
        kwargs["run_first_targets_and_proposals"]((selected,))
        events.append(("runtime_prepare",))
        return prepared_runtime

    step = _load_engine_method(
        "step",
        {
            "build_engine_speculative_partition": (
                lambda *args, **kwargs: partition
            ),
            "build_model_runner_proposal_provider": (
                build_proposal_provider
            ),
            "prepare_native_speculative_batch": prepare_runtime,
            "rollback_prepared_native_speculative_batch": (
                lambda **kwargs: events.append(
                    ("runtime_rollback",)
                )
            ),
            "build_engine_prepared_speculative_commit_rows": (
                lambda *args, **kwargs: (commit_row,)
            ),
            "run_model_runner_tail_batch": (
                lambda *args, **kwargs: ()
            ),
            "ScheduledOutputRow": _ScheduledOutputRow,
        },
    )
    engine = SimpleNamespace(
        _clock_ns=_Clock(10, 20),
        scheduler=Scheduler(),
        model_runner=ModelRunner(),
        speculative_runtime=EngineSpeculativeRuntime(
            draft_adapter=_Adapter(),
            lifecycle=Lifecycle(),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
        flush_pending_hybrid_state_releases=(
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "mixed path must carry releases on ordinary run"
                )
            )
        ),
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )
    engine._kv_offload_identity_rows = (
        lambda seqs:
        build_kv_block_identity_rows(
            engine.scheduler.block_manager,
            tuple(seqs),
        )
    )

    assert step(engine) == ([], 2)
    assert selected.completion_token_ids == [11, 12, 13]
    assert suppressed.completion_token_ids == [21]
    assert events[0:4] == [
        ("drain_releases",),
        (
            "ordinary_run",
            (
                (suppressed,),
                False,
                True,
                "mixed",
                ("lease",),
                (
                    KVBlockIdentityRow(
                        8,
                        ((8, 108),),
                    ),
                ),
            ),
        ),
        (
            "first_targets",
            (
                KVBlockIdentityRow(
                    7,
                    ((7, 107),),
                ),
            ),
        ),
        ("runtime_prepare",),
    ]
    assert events[-3:] == [
        ("scheduler_commit",),
        ("synchronize", 7, (1, 2, 11, 12, 13)),
        ("synchronize", 8, (1, 2, 21)),
    ]
    assert engine.last_step_observation[
        "speculative_output_token_counts"
    ] == {7: 3}
    assert engine.last_step_observation[
        "speculative_accepted_draft_token_counts"
    ] == {7: 2}
    assert engine.last_step_observation[
        "speculative_proposal_token_counts"
    ] == {7: 3}
    assert engine.last_step_observation[
        "speculative_proposal_row_count"
    ] == 1
    assert engine.last_step_observation[
        "speculative_first_target_callback_count"
    ] == 1
    assert engine.last_step_observation[
        "speculative_fixed_q_group_count"
    ] == 1
    timing_ms = engine.last_step_observation[
        "speculative_runtime_timing_ms"
    ]
    assert set(timing_ms) == {
        "draft_proposal_ms",
        "commit_metadata_ms",
    }
    assert timing_ms["draft_proposal_ms"] == 1.0
    assert timing_ms["commit_metadata_ms"] >= 0.0


def _run_selected_step_with_transaction(
    *,
    lifecycle=None,
    kv_commit_error=None,
    scheduler_commit_error=None,
    finished=False,
    model_runner_executor=False,
    proposal_release_error=None,
    residency=False,
    sequence_epoch=0,
    verifier_error=None,
    residency_phase_errors=None,
):
    events = []

    class Sequence:
        seq_id = 7
        num_prompt_tokens = 2
        prefill_chunk_start = 0
        prefill_chunk_end = 0
        prefill_chunk_final = False
        step_is_decode = True
        step_do_sample = True
        status = "running"

        def __init__(self):
            self.token_ids = [1, 2]
            self.block_table = [1]
            self.block_size = 4
            self.finished = False
            self.sequence_epoch = sequence_epoch

        @property
        def completion_token_ids(self):
            return self.token_ids[self.num_prompt_tokens:]

        @property
        def is_finished(self):
            return self.finished

    sequence = Sequence()
    transaction = SimpleNamespace(
        state="materialized",
        original_block_table=(1,),
        original_block_generations=(4,),
        reserved_block_ids=(2,),
        reserved_block_generations=(9,),
    )
    prepared_runtime = SimpleNamespace(
        sequences=(
            SimpleNamespace(
                sequence_id=7,
                sequence=sequence,
                transaction=transaction,
                accepted_tokens=(11, 12),
                proposal=SimpleNamespace(
                    token_ids=(11, 12, 14),
                ),
            ),
        ),
        timing_ms={},
        first_target_callback_count=1,
        tail_callback_count=1,
        side_state_callbacks=None,
        side_state_state="disabled",
        state="prepared",
    )
    commit_row = SimpleNamespace(
        sequence_id=7,
        output_tokens=(11, 12, 13),
        accepted_draft_tokens=(11, 12),
    )
    plan = SimpleNamespace(
        sequence_id=7,
        transaction=transaction,
        committed_block_ids=(2,),
        unused_block_ids=(),
        materialized_end=4,
    )
    class Journal:
        def extend_speculative_kv_plans(
            self,
            scheduler,
            plans,
        ):
            assert scheduler.block_manager is block_manager
            assert plans == (plan,)
            events.append(("scheduler_journal_extend", plans))

    prepared_scheduler = SimpleNamespace(
        state="prepared",
        snapshot=Journal(),
    )

    class BlockManager:
        def prepare_speculative_kv_commit(
            self,
            row_transaction,
            row_sequence,
            accepted_tokens,
        ):
            assert (
                row_transaction,
                row_sequence,
                accepted_tokens,
            ) == (
                transaction,
                sequence,
                (11, 12),
            )
            events.append(("kv_prepare",))
            return plan

        def commit_speculative_kv_commit_batch(self, plans):
            assert plans == (plan,)
            events.append(("kv_commit",))
            if kv_commit_error is not None:
                raise kv_commit_error
            transaction.state = "committed"

    block_manager = BlockManager()

    class Scheduler:
        last_policy_branch = "decode"
        last_speculative_selection = object()
        schedule_generation = 5
        eos = -1

        def __init__(self):
            self.block_manager = block_manager

        def observation_snapshot(self):
            return {"running_seq_ids": [7]}

        def schedule(self, decision_now_ns):
            return [sequence], False, True

        def prepare_postprocess(self, *args, **kwargs):
            events.append(("scheduler_prepare",))
            return prepared_scheduler

        def commit_prepared_postprocess(self, prepared):
            events.append(("scheduler_commit",))
            if scheduler_commit_error is not None:
                prepared.state = (
                    "rollback_failed"
                    if isinstance(
                        scheduler_commit_error,
                        SchedulerPostprocessRollbackError,
                    )
                    else "commit_failed"
                )
                raise scheduler_commit_error
            sequence.token_ids.extend((11, 12, 13))
            sequence.finished = finished
            prepared.state = "committed"

        def last_slo_observation(self):
            return {}

    partition = SimpleNamespace(
        schedule_generation=5,
        selected_sequence_ids=(7,),
        suppressed_sequence_ids=(),
        selected_sequences=(sequence,),
        suppressed_sequences=(),
    )

    def rollback_runtime(**kwargs):
        assert transaction.state == "materialized"
        transaction.state = "rolled_back"
        prepared_runtime.state = "rolled_back"
        events.append(("runtime_rollback",))

    tail_item = SimpleNamespace(
        sequence_id=7,
        plan=SimpleNamespace(logical_slots=(2, 3)),
        proxy_block_table=(1, 2),
        original_block_identities=((1, 4),),
        reserved_block_identities=((2, 9),),
    )

    def prepare_runtime(**kwargs):
        if residency:
            try:
                kwargs["run_tail_batch"]((tail_item,))
            except BaseException:
                transaction.state = "rolled_back"
                events.append(("runtime_rollback",))
                raise
        return prepared_runtime

    def run_tail_batch(model_runner, items, ticket_id=None):
        assert items == (tail_item,)
        assert ticket_id == 41
        events.append(("verifier", ticket_id))
        if verifier_error is not None:
            raise verifier_error
        return ()

    engine_method_namespace = {
        "build_engine_speculative_partition": (
            lambda *args, **kwargs: partition
        ),
        "prepare_native_speculative_batch": prepare_runtime,
        "rollback_prepared_native_speculative_batch": (
            rollback_runtime
        ),
        "build_engine_prepared_speculative_commit_rows": (
            lambda *args, **kwargs: (commit_row,)
        ),
        "run_model_runner_first_targets": (
            lambda *args, **kwargs: ()
        ),
        "run_model_runner_tail_batch": run_tail_batch,
        "build_speculative_residency_prepare_rows": (
            lambda items: (
                SimpleNamespace(
                    sequence_id=7,
                    reserved_block_identities=((2, 9),),
                ),
            )
        ),
        "build_speculative_residency_precommit_rows": (
            lambda plans: (
                SimpleNamespace(
                    sequence_id=7,
                    committed_block_identities=((2, 9),),
                    rejected_block_identities=(),
                ),
            )
        ),
        "ScheduledOutputRow": _ScheduledOutputRow,
    }
    if model_runner_executor:
        finalize_row = SimpleNamespace(
            sequence_id=7,
            proposal_transaction_id="proposal-tx-7",
            accepted_proposal_tokens=2,
        )

        def release_proposal_sequence(
            model_runner,
            descriptor,
            sequence_id,
            row_sequence_epoch,
            *,
            dispatch,
        ):
            assert descriptor.executor_id == "fixture-executor"
            assert callable(dispatch)
            events.append((
                "proposal_release",
                sequence_id,
                row_sequence_epoch,
            ))
            if proposal_release_error is not None:
                raise proposal_release_error

        engine_method_namespace.update({
            "build_prepared_proposal_finalize_rows": (
                lambda prepared: (finalize_row,)
            ),
            "prepare_model_runner_proposal_finalize_batch": (
                lambda *args, **kwargs: (
                    events.append(("proposal_finalize_prepare",))
                    or "proposal-ticket-7"
                )
            ),
            "commit_model_runner_proposal_finalize_batch": (
                lambda *args, **kwargs: events.append(
                    ("proposal_finalize_commit",)
                )
            ),
            "rollback_model_runner_proposal_finalize_batch": (
                lambda *args, **kwargs: events.append(
                    ("proposal_finalize_rollback",)
                )
            ),
            "apply_prepared_speculative_side_state": (
                lambda prepared: (
                    events.append(("side_apply",)),
                    setattr(
                        prepared,
                        "side_state_state",
                        "applied",
                    ),
                )[-1]
            ),
            "seal_prepared_speculative_side_state": (
                lambda prepared: (
                    events.append(("side_seal",)),
                    setattr(
                        prepared,
                        "side_state_state",
                        "sealed",
                    ),
                )[-1]
            ),
            "release_model_runner_proposal_sequence": (
                release_proposal_sequence
            ),
        })

    step = _load_engine_method(
        "step",
        engine_method_namespace,
    )
    model_runner = SimpleNamespace(
        call=lambda *args, **kwargs: None,
        memory_snapshot=lambda: {},
        config=SimpleNamespace(
            kv_offload_mvp0=residency
        ),
        world_size=1,
    )
    engine = SimpleNamespace(
        _clock_ns=_Clock(10, 20),
        scheduler=Scheduler(),
        model_runner=model_runner,
        speculative_runtime=(
            EngineSpeculativeRuntime(
                model_runner_executor=(
                    _model_runner_descriptor(
                        requires_target_hidden=False,
                        requires_proposal_lifecycle=True,
                    )
                ),
                lifecycle=lifecycle,
            )
            if model_runner_executor
            else EngineSpeculativeRuntime(
                draft_adapter=_Adapter(),
                lifecycle=lifecycle,
            )
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
        flush_pending_hybrid_state_releases=(
            lambda **kwargs: events.append(("flush",))
        ),
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )
    if residency:
        engine._speculative_residency_ticket_ids = iter((41,))
        phase_name_by_method = {
            "prepare_speculative_residency_batch": (
                "residency_prepare"
            ),
            "precommit_speculative_residency_batch": (
                "residency_precommit"
            ),
            "rollback_speculative_residency_batch": (
                "residency_rollback"
            ),
            "seal_speculative_residency_batch": (
                "residency_seal"
            ),
        }

        def call_residency_phase(
            method_name,
            ticket_id,
            payload=None,
            **kwargs,
        ):
            phase_name = phase_name_by_method[method_name]
            events.append((phase_name,))
            errors = residency_phase_errors or {}
            error = errors.get(phase_name)
            if error is not None:
                raise error
            return ()

        engine._call_speculative_residency_phase = (
            call_residency_phase
        )
    return (
        step,
        engine,
        sequence,
        transaction,
        prepared_runtime,
        events,
    )


def test_residency_publication_order_wraps_allocator_and_scheduler():
    step, engine, _, _, _, events = (
        _run_selected_step_with_transaction(
            residency=True,
        )
    )

    step(engine)

    journal_event = events[6]
    assert journal_event[0] == "scheduler_journal_extend"
    assert len(journal_event[1]) == 1
    assert journal_event[1][0].transaction is not None
    assert events[:6] == [
        ("flush",),
        ("residency_prepare",),
        ("verifier", 41),
        ("scheduler_prepare",),
        ("kv_prepare",),
        ("residency_precommit",),
    ]
    assert events[7:] == [
        ("kv_commit",),
        ("scheduler_commit",),
        ("residency_seal",),
    ]


def test_verifier_failure_rolls_back_residency_before_allocator():
    error = RuntimeError("verifier failed")
    step, engine, _, transaction, _, events = (
        _run_selected_step_with_transaction(
            residency=True,
            verifier_error=error,
        )
    )

    with pytest.raises(RuntimeError, match="verifier failed"):
        step(engine)

    assert transaction.state == "rolled_back"
    assert events == [
        ("flush",),
        ("residency_prepare",),
        ("verifier", 41),
        ("residency_rollback",),
        ("runtime_rollback",),
    ]


def test_precommit_failure_rolls_back_residency_and_allocator():
    error = RuntimeError("precommit failed")
    step, engine, _, transaction, prepared, events = (
        _run_selected_step_with_transaction(
            residency=True,
            residency_phase_errors={
                "residency_precommit": error,
            },
        )
    )

    with pytest.raises(RuntimeError, match="precommit failed"):
        step(engine)

    assert transaction.state == "rolled_back"
    assert prepared.state == "rolled_back"
    assert events[-3:] == [
        ("residency_precommit",),
        ("residency_rollback",),
        ("runtime_rollback",),
    ]
    assert ("kv_commit",) not in events


def test_allocator_failure_rolls_back_residency_then_runtime():
    error = RuntimeError("kv publication failed")
    step, engine, _, transaction, prepared, events = (
        _run_selected_step_with_transaction(
            residency=True,
            kv_commit_error=error,
        )
    )

    with pytest.raises(RuntimeError, match="kv publication failed"):
        step(engine)

    assert transaction.state == "rolled_back"
    assert prepared.state == "rolled_back"
    assert events[-3:] == [
        ("kv_commit",),
        ("residency_rollback",),
        ("runtime_rollback",),
    ]


def test_seal_failure_keeps_publication_and_poisons_runtime():
    error = RuntimeError("seal failed")
    step, engine, sequence, transaction, prepared, events = (
        _run_selected_step_with_transaction(
            residency=True,
            residency_phase_errors={
                "residency_seal": error,
            },
        )
    )

    with pytest.raises(RuntimeError, match="seal failed"):
        step(engine)

    assert sequence.completion_token_ids == [11, 12, 13]
    assert transaction.state == "committed"
    assert prepared.state == "committed"
    assert engine.speculative_runtime_poisoned is True
    assert "seal failed" in engine.speculative_runtime_poison_reason
    assert events[-3:] == [
        ("kv_commit",),
        ("scheduler_commit",),
        ("residency_seal",),
    ]
    assert ("residency_rollback",) not in events
    assert ("runtime_rollback",) not in events


def test_residency_rollback_failure_poisons_runtime():
    verifier_error = RuntimeError("verifier failed")
    rollback_error = RuntimeError("rollback failed")
    step, engine, _, transaction, _, events = (
        _run_selected_step_with_transaction(
            residency=True,
            verifier_error=verifier_error,
            residency_phase_errors={
                "residency_rollback": rollback_error,
            },
        )
    )

    with pytest.raises(RuntimeError, match="rollback failed"):
        step(engine)

    assert transaction.state == "rolled_back"
    assert engine.speculative_runtime_poisoned is True
    assert "rollback failed" in (
        engine.speculative_runtime_poison_reason
    )
    assert events[-2:] == [
        ("residency_rollback",),
        ("runtime_rollback",),
    ]


def test_step_rolls_back_committed_kv_when_scheduler_commit_fails():
    error = RuntimeError("scheduler mutation failed")
    (
        step,
        engine,
        sequence,
        transaction,
        prepared_runtime,
        events,
    ) = _run_selected_step_with_transaction(
        scheduler_commit_error=error,
    )

    with pytest.raises(
        RuntimeError,
        match="scheduler mutation failed",
    ):
        step(engine)

    assert sequence.completion_token_ids == []
    assert transaction.state == "rolled_back"
    assert prepared_runtime.state == "rolled_back"
    assert events[-3:] == [
        ("kv_commit",),
        ("scheduler_commit",),
        ("runtime_rollback",),
    ]


def test_step_poisons_runtime_when_scheduler_journal_rollback_fails():
    error = SchedulerPostprocessRollbackError(
        RuntimeError("scheduler mutation failed"),
        RuntimeError("journal restore failed"),
    )
    (
        step,
        engine,
        sequence,
        transaction,
        prepared_runtime,
        events,
    ) = _run_selected_step_with_transaction(
        scheduler_commit_error=error,
    )

    with pytest.raises(
        SchedulerPostprocessRollbackError,
        match="scheduler postprocess rollback failed",
    ):
        step(engine)

    assert sequence.completion_token_ids == []
    assert transaction.state == "rolled_back"
    assert prepared_runtime.state == "rolled_back"
    assert engine.speculative_runtime_poisoned is True
    assert (
        engine.speculative_runtime_poison_reason
        == "scheduler postprocess rollback failed: "
        "journal restore failed"
    )
    assert events[-3:] == [
        ("kv_commit",),
        ("scheduler_commit",),
        ("runtime_rollback",),
    ]


def test_step_rolls_back_reservations_when_kv_commit_fails():
    error = RuntimeError("kv publication failed")
    (
        step,
        engine,
        sequence,
        transaction,
        prepared_runtime,
        events,
    ) = _run_selected_step_with_transaction(
        kv_commit_error=error,
    )

    with pytest.raises(
        RuntimeError,
        match="kv publication failed",
    ):
        step(engine)

    assert sequence.completion_token_ids == []
    assert transaction.state == "rolled_back"
    assert prepared_runtime.state == "rolled_back"
    assert ("scheduler_commit",) not in events
    assert events[-2:] == [
        ("kv_commit",),
        ("runtime_rollback",),
    ]


def test_step_releases_finished_lifecycle_after_synchronization():
    events = []

    class Lifecycle:
        def register_sequence(self, sequence_id, token_ids):
            raise AssertionError

        def synchronize_verified_history(
            self,
            sequence_id,
            token_ids,
        ):
            events.append(("synchronize", sequence_id, token_ids))
            return len(token_ids)

        def release_sequence(self, sequence_id):
            events.append(("release", sequence_id))

    step, engine, sequence, _, _, _ = (
        _run_selected_step_with_transaction(
            lifecycle=Lifecycle(),
            finished=True,
        )
    )

    assert step(engine) == (
        [(7, [11, 12, 13])],
        -1,
    )
    assert events == [
        ("synchronize", 7, (1, 2, 11, 12, 13)),
        ("release", 7),
    ]
    assert sequence.is_finished


def test_step_releases_finished_model_runner_proposal_after_seal():
    step, engine, sequence, _, _, events = (
        _run_selected_step_with_transaction(
            finished=True,
            model_runner_executor=True,
            sequence_epoch=3,
        )
    )

    assert step(engine) == (
        [(7, [11, 12, 13])],
        -1,
    )
    assert sequence.is_finished
    assert [
        event
        for event in events
        if event[0]
        in {
            "proposal_finalize_prepare",
            "side_apply",
            "kv_commit",
            "scheduler_commit",
            "proposal_finalize_commit",
            "side_seal",
            "proposal_release",
        }
    ] == [
        ("proposal_finalize_prepare",),
        ("side_apply",),
        ("kv_commit",),
        ("scheduler_commit",),
        ("proposal_finalize_commit",),
        ("side_seal",),
        ("proposal_release", 7, 3),
    ]


def test_step_does_not_release_active_model_runner_proposal_sequence():
    step, engine, _, _, _, events = (
        _run_selected_step_with_transaction(
            model_runner_executor=True,
        )
    )

    step(engine)

    assert not any(
        event[0] == "proposal_release"
        for event in events
    )


def test_step_does_not_release_host_proposal_sequence():
    step, engine, _, _, _, events = (
        _run_selected_step_with_transaction(
            finished=True,
        )
    )

    step(engine)

    assert not any(
        event[0] == "proposal_release"
        for event in events
    )


def test_step_poisoned_after_model_runner_proposal_release_failure():
    error = RuntimeError("proposal release failed")
    step, engine, sequence, transaction, prepared, events = (
        _run_selected_step_with_transaction(
            finished=True,
            model_runner_executor=True,
            proposal_release_error=error,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="proposal release failed",
    ):
        step(engine)

    assert sequence.completion_token_ids == [11, 12, 13]
    assert sequence.is_finished
    assert transaction.state == "committed"
    assert prepared.state == "committed"
    assert engine.speculative_runtime_poisoned is True
    assert "proposal executor sequence release failed" in (
        engine.speculative_runtime_poison_reason
    )
    assert events[-1] == ("proposal_release", 7, 0)


def test_step_poisoned_after_lifecycle_sync_failure():
    class Lifecycle:
        def register_sequence(self, sequence_id, token_ids):
            raise AssertionError

        def synchronize_verified_history(
            self,
            sequence_id,
            token_ids,
        ):
            raise RuntimeError("draft index stale")

        def release_sequence(self, sequence_id):
            raise AssertionError

    step, engine, sequence, transaction, prepared_runtime, _ = (
        _run_selected_step_with_transaction(
            lifecycle=Lifecycle(),
        )
    )

    with pytest.raises(RuntimeError, match="draft index stale"):
        step(engine)

    assert sequence.completion_token_ids == [11, 12, 13]
    assert transaction.state == "committed"
    assert prepared_runtime.state == "committed"
    assert engine.speculative_runtime_poisoned is True
    assert "draft lifecycle synchronization failed" in (
        engine.speculative_runtime_poison_reason
    )
    engine._clock_ns = _Clock(30)
    engine.flush_pending_hybrid_state_releases = (
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "poisoned runtime must fail before ModelRunner work"
            )
        )
    )
    with pytest.raises(
        RuntimeError,
        match="speculative runtime is poisoned",
    ):
        step(engine)
