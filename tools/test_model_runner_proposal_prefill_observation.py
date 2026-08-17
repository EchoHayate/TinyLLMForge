from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import types

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm")
]
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
sys.modules.setdefault(
    "tinyvllm.speculative",
    speculative_package,
)

from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalExecutorRegistry,
    TargetPrefillObservation,
)
from tinyvllm.speculative.adapter import DraftCapabilities


class _Rows:
    def __init__(self, values):
        self.values = tuple(values)

    @property
    def shape(self):
        if (
            self.values
            and isinstance(self.values[0], tuple)
        ):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _Rows(self.values[key])
        return self.values[key]


class _LifecycleExecutor:
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.observations = []

    def observe_target_prefill(self, rows):
        self.observations.append(rows)

    def propose_batch(self, inputs):
        return ()

    def prepare_finalize_batch(self, rows):
        return "ticket-1"

    def commit_finalize_batch(self, ticket_id):
        return None

    def rollback_finalize_batch(self, ticket_id):
        return None

    def release_sequence(
        self,
        sequence_id,
        *,
        sequence_epoch,
    ):
        return None


@dataclass
class _Sequence:
    seq_id: int
    token_ids: tuple[int, ...]
    prefill_chunk_start: int
    prefill_chunk_end: int
    prefill_chunk_final: bool
    sequence_epoch: int = 0
    hybrid_state_slot_id: int = -1
    hybrid_state_generation: int = 0

    def __len__(self):
        return len(self.token_ids)


def _model_runner_shell():
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
        "_proposal_prefill_observation_required",
        "_observe_proposal_target_prefill",
        "observe_speculative_target_prefill_batch",
        "_run_model_step",
    }
    methods = [
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in method_names
    ]
    shell = ast.ClassDef(
        name="ModelRunnerPrefillObservationShell",
        bases=[],
        keywords=[],
        body=methods or [ast.Pass()],
        decorator_list=[],
    )
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "HybridStateLease": object,
        "TargetPrefillObservation": TargetPrefillObservation,
        "_qwen35_step_token_counts": (
            lambda seqs, **kwargs: ()
        ),
        "_round_qwen35_final_prefill_recurrent_states": (
            lambda *args, **kwargs: None
        ),
        "reset_context": lambda: None,
    }
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
    return namespace["ModelRunnerPrefillObservationShell"]


def _constructed_runner(*, lifecycle):
    runner = _model_runner_shell()()
    registry = ModelRunnerProposalExecutorRegistry()
    executor = None
    if lifecycle:
        capabilities = DraftCapabilities(
            source_type="fixture",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=4,
            execution_domain="model_runner",
            requires_proposal_lifecycle=True,
        )
        executor = _LifecycleExecutor(capabilities)
        registry.register(
            "fixture",
            executor,
            capabilities,
        )
    runner.speculative_proposal_executors = registry
    runner._prepare_hybrid_state_batch = (
        lambda seqs, released: None
    )
    runner._capture_qwen35_recurrent_source_state = (
        lambda *args, **kwargs: None
    )
    runner.hybrid_state_runtime_bridge = None
    runner._kv_offload_before_forward = lambda: None
    runner._kv_offload_after_forward = lambda: None
    runner._last_step_logits_cpu = None
    runner._record_step_logits = False
    runner.rank = 0
    runner.run_model_calls = []

    def prepare_prefill(seqs):
        token_ids = []
        positions = []
        for seq in seqs:
            token_ids.extend(
                seq.token_ids[
                    seq.prefill_chunk_start:
                    seq.prefill_chunk_end
                ]
            )
            positions.extend(
                range(
                    seq.prefill_chunk_start,
                    seq.prefill_chunk_end,
                )
            )
        return _Rows(token_ids), _Rows(positions)

    def run_model(
        input_ids,
        positions,
        is_prefill,
        **kwargs,
    ):
        runner.run_model_calls.append(
            (input_ids, positions, is_prefill, kwargs)
        )
        logits = _Rows((0,) * len(input_ids.values))
        if kwargs.get("return_hidden", False):
            hidden = _Rows(
                tuple(
                    (index, index, index, index)
                    for index in range(len(input_ids.values))
                )
            )
            return logits, hidden
        return logits

    runner.prepare_prefill = prepare_prefill
    runner.prepare_decode = lambda seqs: (_Rows(()), _Rows(()))
    runner.run_model = run_model
    return runner, executor


def test_final_prefill_returns_hidden_only_for_lifecycle_executor():
    runner, executor = _constructed_runner(lifecycle=True)
    seq = _Sequence(
        seq_id=7,
        token_ids=(10, 11, 12),
        prefill_chunk_start=0,
        prefill_chunk_end=3,
        prefill_chunk_final=True,
    )

    result = runner._run_model_step(
        [seq],
        True,
        False,
    )

    observation = executor.observations[0][0]
    assert result is None
    assert observation.sequence_id == 7
    assert observation.token_ids == (10, 11, 12)
    assert observation.sequence_epoch == 0
    assert observation.is_final_chunk is True
    assert observation.positions.values == (0, 1, 2)
    assert observation.target_hidden.shape == (3, 4)
    assert (
        runner.run_model_calls[0][3]["return_hidden"]
        is True
    )


def test_prefill_observation_slices_each_exact_chunk():
    runner, executor = _constructed_runner(lifecycle=True)
    seqs = [
        _Sequence(
            seq_id=7,
            token_ids=(10, 11, 12),
            prefill_chunk_start=0,
            prefill_chunk_end=2,
            prefill_chunk_final=False,
        ),
        _Sequence(
            seq_id=8,
            token_ids=(20, 21, 22, 23),
            prefill_chunk_start=1,
            prefill_chunk_end=4,
            prefill_chunk_final=True,
        ),
    ]

    runner._run_model_step(seqs, True, False)

    rows = executor.observations[0]
    assert tuple(row.sequence_id for row in rows) == (7, 8)
    assert rows[0].token_ids == (10, 11)
    assert rows[0].positions.values == (0, 1)
    assert rows[0].target_hidden.shape == (2, 4)
    assert rows[0].is_final_chunk is False
    assert rows[1].token_ids == (21, 22, 23)
    assert rows[1].positions.values == (1, 2, 3)
    assert rows[1].target_hidden.shape == (3, 4)
    assert rows[1].is_final_chunk is True


def test_prefill_observation_is_delivered_to_two_lifecycle_sources():
    runner, first_executor = _constructed_runner(lifecycle=True)
    capabilities = DraftCapabilities(
        source_type="independent_draft_model",
        supports_batch=True,
        requires_target_hidden=False,
        requires_target_logits=False,
        max_proposal_tokens=4,
        execution_domain="model_runner",
        requires_proposal_lifecycle=True,
    )
    second_executor = _LifecycleExecutor(capabilities)
    runner.speculative_proposal_executors.register(
        "autoregressive-draft",
        second_executor,
        capabilities,
    )
    seq = _Sequence(
        seq_id=9,
        token_ids=(30, 31),
        prefill_chunk_start=0,
        prefill_chunk_end=2,
        prefill_chunk_final=True,
    )

    runner._run_model_step([seq], True, False)

    assert first_executor.observations[0][0].token_ids == (30, 31)
    assert second_executor.observations[0][0].token_ids == (30, 31)


def test_prefill_without_lifecycle_does_not_request_hidden():
    runner, executor = _constructed_runner(lifecycle=False)
    seq = _Sequence(
        seq_id=7,
        token_ids=(10, 11, 12),
        prefill_chunk_start=0,
        prefill_chunk_end=3,
        prefill_chunk_final=True,
    )

    result = runner._run_model_step(
        [seq],
        True,
        False,
    )

    assert executor is None
    assert result is None
    assert (
        runner.run_model_calls[0][3].get(
            "return_hidden",
            False,
        )
        is False
    )


def test_prefill_observation_rejects_mixed_batch():
    runner, _ = _constructed_runner(lifecycle=True)
    seq = _Sequence(
        seq_id=7,
        token_ids=(10, 11, 12),
        prefill_chunk_start=0,
        prefill_chunk_end=2,
        prefill_chunk_final=False,
    )
    seq.step_is_decode = False

    with pytest.raises(ValueError, match="mixed"):
        runner._run_model_step(
            [seq],
            True,
            False,
            batch_kind="mixed",
        )

    assert runner.run_model_calls == []
