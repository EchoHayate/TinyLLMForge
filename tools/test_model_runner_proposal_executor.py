from __future__ import annotations

import ast
from dataclasses import replace
import os
import sys
import types
from types import SimpleNamespace

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "engine")
]
speculative_package = types.ModuleType("tinyvllm.speculative")
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)
sys.modules.setdefault("tinyvllm.speculative", speculative_package)

import tinyvllm.engine.speculative_proposal_executor as proposal_executor_module
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalExecutorRegistry,
    ModelRunnerProposalInput,
    assert_tensor_free,
)
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)


class _Executor:
    def __init__(
        self,
        capabilities,
        proposals=(),
        error=None,
    ):
        self.capabilities = capabilities
        self.proposals = tuple(proposals)
        self.error = error
        self.calls = []

    def propose_batch(self, inputs):
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        return self.proposals


class _LifecycleExecutor(_Executor):
    def __init__(self, capabilities, proposals=()):
        super().__init__(capabilities, proposals)
        self.observations = []
        self.finalize_events = []
        self.release_events = []
        self.release_result = None
        self.next_ticket = "ticket-1"

    def observe_target_prefill(self, rows):
        self.observations.append(rows)

    def prepare_finalize_batch(self, rows):
        self.finalize_events.append(("prepare", rows))
        return self.next_ticket

    def commit_finalize_batch(self, ticket_id):
        self.finalize_events.append(("commit", ticket_id))

    def rollback_finalize_batch(self, ticket_id):
        self.finalize_events.append(("rollback", ticket_id))

    def release_sequence(
        self,
        sequence_id,
        *,
        sequence_epoch,
    ):
        self.release_events.append(
            (sequence_id, sequence_epoch)
        )
        return self.release_result


def _capabilities(**overrides):
    values = {
        "source_type": "fixture",
        "supports_batch": True,
        "requires_target_hidden": False,
        "requires_target_logits": False,
        "max_proposal_tokens": 4,
        "execution_domain": "model_runner",
    }
    values.update(overrides)
    return DraftCapabilities(**values)


def _input(sequence_id, **overrides):
    values = {
        "sequence_id": sequence_id,
        "token_ids": (1, 2, sequence_id),
        "remaining_output_tokens": 4,
        "max_proposal_tokens": 4,
        "first_target_token": 100 + sequence_id,
        "target_hidden": None,
        "target_logits": None,
    }
    values.update(overrides)
    return ModelRunnerProposalInput(**values)


def _proposal(sequence_id, token_ids, **overrides):
    values = {
        "sequence_id": sequence_id,
        "token_ids": tuple(token_ids),
        "source_type": "fixture",
        "metadata": None,
        "timing_ms": {"draft_ms": 0.25},
    }
    values.update(overrides)
    return DraftProposal(**values)


def _observation(sequence_id=1, **overrides):
    values = {
        "sequence_id": sequence_id,
        "sequence_epoch": 0,
        "token_ids": (1, 2),
        "positions": object(),
        "target_hidden": object(),
        "is_final_chunk": True,
    }
    values.update(overrides)
    return proposal_executor_module.TargetPrefillObservation(
        **values
    )


def _finalize_row(sequence_id=1, **overrides):
    values = {
        "sequence_id": sequence_id,
        "proposal_transaction_id": f"tx-{sequence_id}",
        "accepted_proposal_tokens": 2,
    }
    values.update(overrides)
    return proposal_executor_module.ProposalFinalizeRow(
        **values
    )


def _model_runner_lifecycle_shell():
    source_path = os.path.join(
        _REPO_ROOT,
        "tinyvllm",
        "engine",
        "model_runner.py",
    )
    with open(source_path, encoding="utf-8") as source_file:
        module = ast.parse(source_file.read())
    model_runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method_names = {
        "observe_speculative_target_prefill_batch",
        "prepare_speculative_proposal_finalize_batch",
        "commit_speculative_proposal_finalize_batch",
        "rollback_speculative_proposal_finalize_batch",
        "release_speculative_proposal_sequence",
    }
    methods = [
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in method_names
    ]
    shell = ast.ClassDef(
        name="ModelRunnerLifecycleShell",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[shell],
                    type_ignores=[],
                )
            ),
            source_path,
            "exec",
        ),
        namespace,
    )
    return namespace["ModelRunnerLifecycleShell"]


def test_registry_executes_once_and_restores_input_order():
    capabilities = _capabilities()
    executor = _Executor(
        capabilities,
        (
            _proposal(2, ()),
            _proposal(1, (11, 12)),
        ),
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture-executor",
        executor,
        capabilities,
    )
    inputs = (_input(1), _input(2))

    rows = registry.execute_batch(
        "fixture-executor",
        inputs,
        capabilities,
    )

    assert tuple(row.sequence_id for row in rows) == (1, 2)
    assert rows[0].token_ids == (11, 12)
    assert rows[1].token_ids == ()
    assert executor.calls == [inputs]


def test_compact_worker_context_uses_explicit_sequence_token_count():
    capabilities = _capabilities(
        requires_full_token_history=False,
    )
    sequence = SimpleNamespace(num_tokens=4097)
    normalize = getattr(
        proposal_executor_module,
        "model_runner_proposal_token_context",
    )

    token_ids, context_token_count = normalize(
        sequence,
        capabilities,
    )

    assert token_ids == ()
    assert context_token_count == 4097


def test_full_history_executor_rejects_decode_worker_compact_sequence():
    capabilities = _capabilities()
    sequence = SimpleNamespace(num_tokens=4097)
    normalize = getattr(
        proposal_executor_module,
        "model_runner_proposal_token_context",
    )

    with pytest.raises(
        RuntimeError,
        match="full token history",
    ):
        normalize(sequence, capabilities)


def test_model_runner_lifecycle_methods_route_registered_capabilities():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(capabilities)
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)
    runner = _model_runner_lifecycle_shell()()
    runner.speculative_proposal_executors = registry
    observation = _observation()
    finalize_row = _finalize_row()

    runner.observe_speculative_target_prefill_batch(
        "fixture",
        (observation,),
    )
    ticket = runner.prepare_speculative_proposal_finalize_batch(
        "fixture",
        (finalize_row,),
    )
    runner.commit_speculative_proposal_finalize_batch(
        "fixture",
        ticket,
    )
    runner.rollback_speculative_proposal_finalize_batch(
        "fixture",
        "ticket-2",
    )
    runner.release_speculative_proposal_sequence(
        "fixture",
        1,
        0,
    )

    assert executor.observations == [(observation,)]
    assert executor.finalize_events == [
        ("prepare", (finalize_row,)),
        ("commit", "ticket-1"),
        ("rollback", "ticket-2"),
    ]
    assert executor.release_events == [(1, 0)]


def test_registry_runs_lifecycle_prefill_and_two_phase_finalize():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(
        capabilities,
        (
            _proposal(
                1,
                (101, 102),
                proposal_transaction_id="tx-1",
            ),
        ),
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)
    observation = _observation()
    finalize_row = _finalize_row()

    registry.observe_target_prefill(
        "fixture",
        (observation,),
        capabilities,
    )
    ticket = registry.prepare_finalize_batch(
        "fixture",
        (finalize_row,),
        capabilities,
    )
    registry.commit_finalize_batch(
        "fixture",
        ticket,
        capabilities,
    )

    assert executor.observations == [(observation,)]
    assert ticket == "ticket-1"
    assert executor.finalize_events == [
        ("prepare", (finalize_row,)),
        ("commit", "ticket-1"),
    ]


def test_registry_rolls_back_prepared_lifecycle_ticket():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(capabilities)
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)

    registry.rollback_finalize_batch(
        "fixture",
        "ticket-1",
        capabilities,
    )

    assert executor.finalize_events == [
        ("rollback", "ticket-1"),
    ]


def test_registry_releases_lifecycle_sequence():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(capabilities)
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)

    registry.release_sequence(
        "fixture",
        7,
        3,
        capabilities,
    )

    assert executor.release_events == [(7, 3)]


@pytest.mark.parametrize(
    ("sequence_id", "sequence_epoch", "match"),
    (
        (True, 0, "sequence ID"),
        (-1, 0, "sequence ID"),
        (7, True, "sequence epoch"),
        (7, -1, "sequence epoch"),
    ),
)
def test_registry_rejects_invalid_release_identity(
    sequence_id,
    sequence_epoch,
    match,
):
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match=match):
        registry.release_sequence(
            "fixture",
            sequence_id,
            sequence_epoch,
            capabilities,
        )


def test_registry_rejects_non_none_release_acknowledgement():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(capabilities)
    executor.release_result = "unexpected"
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)

    with pytest.raises(ValueError, match="acknowledgement"):
        registry.release_sequence(
            "fixture",
            7,
            0,
            capabilities,
        )


def test_registry_rejects_lifecycle_capability_mismatch():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="capabilities"):
        registry.observe_target_prefill(
            "fixture",
            (_observation(),),
            replace(
                capabilities,
                max_proposal_tokens=3,
            ),
        )


def test_lifecycle_proposal_requires_transaction_id():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(
            capabilities,
            (_proposal(1, (101, 102)),),
        ),
        capabilities,
    )

    with pytest.raises(ValueError, match="transaction"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            capabilities,
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"proposal_transaction_id": ""},
            "transaction",
        ),
        (
            {"accepted_proposal_tokens": True},
            "accepted",
        ),
        (
            {"accepted_proposal_tokens": -1},
            "accepted",
        ),
    ],
)
def test_registry_rejects_invalid_finalize_rows(
    overrides,
    match,
):
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )
    row = _finalize_row(**overrides)

    with pytest.raises(ValueError, match=match):
        registry.prepare_finalize_batch(
            "fixture",
            (row,),
            capabilities,
        )


@pytest.mark.parametrize(
    "row_overrides,match",
    [
        (
            (
                {"sequence_id": 1},
                {"sequence_id": 1},
            ),
            "sequence",
        ),
        (
            (
                {"sequence_id": 1},
                {
                    "sequence_id": 2,
                    "proposal_transaction_id": "tx-1",
                },
            ),
            "transaction",
        ),
    ],
)
def test_registry_rejects_duplicate_finalize_identity(
    row_overrides,
    match,
):
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )
    rows = tuple(
        _finalize_row(**overrides)
        for overrides in row_overrides
    )

    with pytest.raises(ValueError, match=match):
        registry.prepare_finalize_batch(
            "fixture",
            rows,
            capabilities,
        )


@pytest.mark.parametrize("ticket_id", ["", 1, None])
def test_registry_rejects_invalid_finalize_ticket(ticket_id):
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="ticket"):
        registry.commit_finalize_batch(
            "fixture",
            ticket_id,
            capabilities,
        )


def test_registry_rejects_invalid_prepared_ticket():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    executor = _LifecycleExecutor(capabilities)
    executor.next_ticket = ""
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)

    with pytest.raises(ValueError, match="ticket"):
        registry.prepare_finalize_batch(
            "fixture",
            (_finalize_row(),),
            capabilities,
        )


def test_registry_rejects_lifecycle_call_for_stateless_executor():
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="lifecycle"):
        registry.observe_target_prefill(
            "fixture",
            (_observation(),),
            capabilities,
        )


def test_registry_rejects_duplicate_prefill_sequence_ids():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _LifecycleExecutor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="sequence"):
        registry.observe_target_prefill(
            "fixture",
            (_observation(1), _observation(1)),
            capabilities,
        )


@pytest.mark.parametrize("executor_id", ["", 1, None])
def test_registry_rejects_invalid_executor_id(executor_id):
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()

    with pytest.raises(ValueError, match="executor ID"):
        registry.register(
            executor_id,
            _Executor(capabilities),
            capabilities,
        )


def test_registry_rejects_duplicate_executor_id():
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            "fixture",
            _Executor(capabilities),
            capabilities,
        )


def test_registry_rejects_host_capabilities():
    capabilities = _capabilities(execution_domain="host")
    registry = ModelRunnerProposalExecutorRegistry()

    with pytest.raises(ValueError, match="model_runner"):
        registry.register(
            "fixture",
            _Executor(capabilities),
            capabilities,
        )


def test_registry_rejects_executor_capability_mismatch():
    capabilities = _capabilities()
    executor = _Executor(
        replace(capabilities, max_proposal_tokens=3)
    )
    registry = ModelRunnerProposalExecutorRegistry()

    with pytest.raises(ValueError, match="capabilities"):
        registry.register(
            "fixture",
            executor,
            capabilities,
        )


def test_registry_rejects_missing_executor():
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()

    with pytest.raises(ValueError, match="not registered"):
        registry.execute_batch(
            "missing",
            (_input(1),),
            capabilities,
        )


def test_registry_rejects_requested_capability_mismatch():
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match="capabilities"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            replace(capabilities, max_proposal_tokens=3),
        )


def test_registry_propagates_executor_failure():
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(
            capabilities,
            error=RuntimeError("executor failed"),
        ),
        capabilities,
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            capabilities,
        )


@pytest.mark.parametrize(
    "inputs,match",
    [
        ((), "non-empty"),
        ((_input(1), _input(1)), "unique"),
        ((_input(True),), "integer"),
    ],
)
def test_registry_rejects_invalid_input_identity(inputs, match):
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match=match):
        registry.execute_batch(
            "fixture",
            inputs,
            capabilities,
        )


@pytest.mark.parametrize(
    "capabilities,input_row,match",
    [
        (
            _capabilities(requires_target_hidden=True),
            _input(1, target_hidden=None),
            "hidden",
        ),
        (
            _capabilities(requires_target_logits=True),
            _input(1, target_logits=None),
            "logits",
        ),
    ],
)
def test_registry_rejects_missing_required_target_payload(
    capabilities,
    input_row,
    match,
):
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities),
        capabilities,
    )

    with pytest.raises(ValueError, match=match):
        registry.execute_batch(
            "fixture",
            (input_row,),
            capabilities,
        )


@pytest.mark.parametrize(
    "proposals,match",
    [
        ((_proposal(1, (10,)),), "exactly match"),
        (
            (
                _proposal(1, (10,)),
                _proposal(1, (11,)),
            ),
            "unique",
        ),
        (
            (
                _proposal(1, (10,)),
                _proposal(2, (20,)),
                _proposal(3, (30,)),
            ),
            "exactly match",
        ),
    ],
)
def test_registry_rejects_invalid_proposal_identity(
    proposals,
    match,
):
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities, proposals),
        capabilities,
    )

    with pytest.raises(ValueError, match=match):
        registry.execute_batch(
            "fixture",
            (_input(1), _input(2)),
            capabilities,
        )


@pytest.mark.parametrize(
    "proposal,match",
    [
        (_proposal(1, (1, 2, 3, 4, 5)), "capability"),
        (
            _proposal(1, (1, 2, 3)),
            "input limit",
        ),
        (
            _proposal(1, (1,), source_type="other"),
            "source_type",
        ),
        (
            _proposal(1, (True,)),
            "token",
        ),
    ],
)
def test_registry_rejects_invalid_proposal_payload(
    proposal,
    match,
):
    capabilities = _capabilities()
    input_row = (
        _input(1, max_proposal_tokens=2)
        if match == "input limit"
        else _input(1)
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(capabilities, (proposal,)),
        capabilities,
    )

    with pytest.raises(ValueError, match=match):
        registry.execute_batch(
            "fixture",
            (input_row,),
            capabilities,
        )


def test_registry_rejects_tensor_in_public_proposal():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    capabilities = _capabilities()
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(
            capabilities,
            (
                _proposal(
                    1,
                    (10,),
                    metadata={"hidden": Tensor()},
                ),
            ),
        ),
        capabilities,
    )

    with pytest.raises(ValueError, match="proposal.*tensor"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            capabilities,
        )


def test_assert_tensor_free_accepts_nested_plain_values():
    value = {
        "rows": (
            {"tokens": [1, 2]},
            {"metadata": {"draft_ms": 0.25}},
        )
    }

    assert_tensor_free(value, name="result")


def test_assert_tensor_free_rejects_nested_tensor():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    value = {"rows": [{"metadata": (Tensor(),)}]}

    with pytest.raises(ValueError, match="result.*tensor"):
        assert_tensor_free(value, name="result")


def test_assert_tensor_free_handles_recursive_container():
    value = []
    value.append(value)

    assert_tensor_free(value, name="result")
