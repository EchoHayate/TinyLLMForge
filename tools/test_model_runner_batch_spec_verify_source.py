from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_RUNNER_PATH = (
    _REPO_ROOT / "tinyvllm" / "engine" / "model_runner.py"
)


def _method():
    tree = ast.parse(_MODEL_RUNNER_PATH.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    return next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_spec_verify_batch"
    )


def _attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _parent_map(root):
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


def _contains(root, target):
    return any(node is target for node in ast.walk(root))


def test_batch_verifier_uses_one_prepare_and_one_model_forward():
    method = _method()
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    ]

    assert sum(
        _attribute_path(call.func)
        == ("self", "prepare_spec_verify_batch")
        for call in calls
    ) == 1
    run_calls = [
        call
        for call in calls
        if _attribute_path(call.func) == ("self", "run_model")
    ]
    assert len(run_calls) == 2
    assert any(
        isinstance(node, ast.If)
        and any(_contains(row, run_calls[0]) for row in node.body)
        and any(_contains(row, run_calls[1]) for row in node.orelse)
        for node in ast.walk(method)
    )
    assert all(
        _attribute_path(call.func)
        != ("self", "prepare_spec_verify")
        for call in calls
    )


def test_model_forward_is_not_nested_in_per_item_iteration():
    method = _method()
    parents = _parent_map(method)
    run_calls = [
        node
        for node in ast.walk(method)
        if (
            isinstance(node, ast.Call)
            and _attribute_path(node.func)
            == ("self", "run_model")
        )
    ]
    forbidden = (
        ast.For,
        ast.AsyncFor,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    for run_call in run_calls:
        ancestor = parents.get(run_call)
        while ancestor is not None:
            assert not isinstance(ancestor, forbidden)
            ancestor = parents.get(ancestor)


def test_context_reset_is_in_finally_block():
    method = _method()
    try_nodes = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Try)
    ]

    reset_finally_nodes = [
        try_node
        for try_node in try_nodes
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reset_context"
            for statement in try_node.finalbody
            for node in ast.walk(statement)
        )
    ]
    assert len(reset_finally_nodes) == 1
