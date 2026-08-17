from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"


def _load_publication_helper(namespace):
    tree = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(ENGINE_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_commit_prepared_speculative_publication"
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[
        "_commit_prepared_speculative_publication"
    ]


def _fixture(fail_at=None):
    events = []
    live_side_state = {"value": "original"}
    prepared = SimpleNamespace(
        state="prepared",
        side_state_state="selected",
        side_state_callbacks=object(),
    )
    transaction = SimpleNamespace(state="materialized")
    plan = SimpleNamespace(transaction=transaction)
    engine = SimpleNamespace(
        model_runner=object(),
        scheduler=SimpleNamespace(
            block_manager=SimpleNamespace(
                commit_speculative_kv_commit_batch=lambda plans: (
                    (_ for _ in ()).throw(
                        RuntimeError("KV commit failed")
                    )
                    if fail_at == "kv_commit"
                    else (
                        events.append("kv_commit"),
                        setattr(transaction, "state", "committed"),
                    )
                )
            ),
            commit_prepared_postprocess=lambda value: (
                (_ for _ in ()).throw(
                    RuntimeError("Scheduler commit failed")
                )
                if fail_at == "scheduler_commit"
                else events.append("scheduler_commit")
            ),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
    )
    finalize_rows = (
        SimpleNamespace(
            sequence_id=8,
            proposal_transaction_id="proposal-8",
            accepted_proposal_tokens=2,
        ),
    )

    def prepare_finalize(*_args):
        if fail_at == "finalize_prepare":
            raise RuntimeError("finalize prepare failed")
        events.append("finalize_prepare")
        return "ticket-1"

    def commit_finalize(*_args):
        if fail_at == "finalize_commit":
            raise RuntimeError("finalize commit failed")
        events.append("finalize_commit")

    def rollback_finalize(*_args):
        events.append("finalize_rollback")

    def apply_side(actual):
        if fail_at == "side_apply":
            raise RuntimeError("side apply failed")
        events.append("side_apply")
        live_side_state["value"] = "selected"
        actual.side_state_state = "applied"

    def seal_side(actual):
        if fail_at == "side_seal":
            raise RuntimeError("side seal failed")
        events.append("side_seal")
        actual.side_state_state = "sealed"

    def rollback_side(actual):
        events.append("side_rollback")
        live_side_state["value"] = "original"
        actual.side_state_state = "rolled_back"

    helper = _load_publication_helper({
        "build_prepared_proposal_finalize_rows": (
            lambda actual: finalize_rows
        ),
        "prepare_model_runner_proposal_finalize_batch": (
            prepare_finalize
        ),
        "commit_model_runner_proposal_finalize_batch": (
            commit_finalize
        ),
        "rollback_model_runner_proposal_finalize_batch": (
            rollback_finalize
        ),
        "apply_prepared_speculative_side_state": apply_side,
        "seal_prepared_speculative_side_state": seal_side,
        "rollback_prepared_speculative_side_state": (
            rollback_side
        ),
    })
    runtime = SimpleNamespace(
        model_runner_executor=SimpleNamespace(
            executor_id="fixture"
        )
    )
    return (
        helper,
        engine,
        runtime,
        prepared,
        (plan,),
        events,
        live_side_state,
        transaction,
    )


def test_side_state_success_publication_order():
    (
        helper,
        engine,
        runtime,
        prepared,
        plans,
        events,
        live_side_state,
        _,
    ) = _fixture()

    helper(
        engine,
        runtime,
        prepared,
        plans,
        object(),
    )

    assert events == [
        "finalize_prepare",
        "side_apply",
        "kv_commit",
        "scheduler_commit",
        "finalize_commit",
        "side_seal",
    ]
    assert prepared.state == "committed"
    assert prepared.side_state_state == "sealed"
    assert live_side_state["value"] == "selected"
    assert engine.speculative_runtime_poisoned is False


@pytest.mark.parametrize(
    "fail_at",
    (
        "side_apply",
        "kv_commit",
        "scheduler_commit",
    ),
)
def test_side_state_previsibility_failure_restores_original(
    fail_at,
):
    (
        helper,
        engine,
        runtime,
        prepared,
        plans,
        events,
        live_side_state,
        transaction,
    ) = _fixture(fail_at)

    with pytest.raises(RuntimeError, match="failed"):
        helper(
            engine,
            runtime,
            prepared,
            plans,
            object(),
        )

    assert events[-2:] == [
        "finalize_rollback",
        "side_rollback",
    ]
    assert live_side_state["value"] == "original"
    assert transaction.state == "materialized"
    assert engine.speculative_runtime_poisoned is False


def test_side_state_finalize_prepare_failure_does_not_apply():
    (
        helper,
        engine,
        runtime,
        prepared,
        plans,
        events,
        live_side_state,
        _,
    ) = _fixture("finalize_prepare")

    with pytest.raises(RuntimeError, match="finalize prepare failed"):
        helper(
            engine,
            runtime,
            prepared,
            plans,
            object(),
        )

    assert events == []
    assert live_side_state["value"] == "original"
    assert prepared.side_state_state == "selected"


@pytest.mark.parametrize(
    ("fail_at", "reason"),
    (
        (
            "finalize_commit",
            "proposal finalization commit failed",
        ),
        (
            "side_seal",
            "speculative side-state seal failed",
        ),
    ),
)
def test_side_state_postvisibility_failure_poisoned(
    fail_at,
    reason,
):
    (
        helper,
        engine,
        runtime,
        prepared,
        plans,
        events,
        live_side_state,
        _,
    ) = _fixture(fail_at)

    with pytest.raises(RuntimeError, match="failed"):
        helper(
            engine,
            runtime,
            prepared,
            plans,
            object(),
        )

    assert engine.speculative_runtime_poisoned is True
    assert reason in engine.speculative_runtime_poison_reason
    assert prepared.state == "committed"
    assert live_side_state["value"] == "selected"
    assert "side_rollback" not in events
