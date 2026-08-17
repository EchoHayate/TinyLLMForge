from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LLM_ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"
SNAPSHOT_FIELDS = {
    "rank",
    "world_size",
    "registered",
    "registration_consensus_sha256",
    "executor_descriptor",
    "checkpoint_identity",
    "tokenizer_contract",
    "registration_error",
    "executor",
}


def _load_method():
    tree = ast.parse(
        LLM_ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_ENGINE_PATH),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "autoregressive_draft_authority_snapshots"
        ),
        None,
    )
    assert method_node is not None, (
        "LLMEngine.autoregressive_draft_authority_snapshots "
        "is missing"
    )
    function = ast.FunctionDef(
        name=method_node.name,
        args=method_node.args,
        body=method_node.body,
        decorator_list=[],
        returns=method_node.returns,
        type_comment=method_node.type_comment,
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(LLM_ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[method_node.name]


def _snapshot(rank, *, world_size=4):
    return {
        "rank": rank,
        "world_size": world_size,
        "registered": True,
        "registration_consensus_sha256": "a" * 64,
        "executor_descriptor": {"executor_id": "autoregressive-draft"},
        "checkpoint_identity": {"target": {}, "draft": {}},
        "tokenizer_contract": {"target": {}, "draft": {}},
        "registration_error": None,
        "executor": {"rank": rank, "world_size": world_size},
    }


def _engine(local, worker_rows, *, world_size=4):
    calls = []

    def call(method_name, *, timeout_s):
        calls.append((method_name, timeout_s))
        acknowledgements = tuple(
            SimpleNamespace(rank=rank, result=row)
            for rank, row in worker_rows
        )
        return local, acknowledgements

    return (
        SimpleNamespace(
            call_model_runner_acknowledged=call,
            model_runner=SimpleNamespace(world_size=world_size),
        ),
        calls,
    )


def test_engine_collects_exact_learned_drafter_rank_inventory():
    snapshots = _load_method()
    engine, calls = _engine(
        _snapshot(0),
        tuple((rank, _snapshot(rank)) for rank in (1, 2, 3)),
    )

    rows = snapshots(engine, timeout_s=12.0)

    assert tuple(row["rank"] for row in rows) == (0, 1, 2, 3)
    assert all(set(row) == SNAPSHOT_FIELDS for row in rows)
    assert calls == [
        ("autoregressive_draft_authority_snapshot", 12.0)
    ]


@pytest.mark.parametrize(
    ("local", "worker_rows", "message"),
    (
        (
            _snapshot(0),
            (
                (1, _snapshot(2)),
                (2, _snapshot(1)),
                (3, _snapshot(3)),
            ),
            "rank mismatch",
        ),
        (
            _snapshot(0),
            (
                (1, _snapshot(1)),
                (3, _snapshot(3)),
            ),
            "rank inventory",
        ),
        (
            _snapshot(0),
            (
                (1, _snapshot(1)),
                (2, _snapshot(2, world_size=3)),
                (3, _snapshot(3)),
            ),
            "world size",
        ),
        (
            {**_snapshot(0), "unexpected": True},
            (
                (1, _snapshot(1)),
                (2, _snapshot(2)),
                (3, _snapshot(3)),
            ),
            "fields",
        ),
    ),
)
def test_engine_rejects_invalid_learned_drafter_rank_transport(
    local,
    worker_rows,
    message,
):
    snapshots = _load_method()
    engine, _ = _engine(local, worker_rows)

    with pytest.raises(ValueError, match=message):
        snapshots(engine, timeout_s=12.0)
