from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_PATH = _REPO_ROOT / "tinyvllm" / "engine" / "llm_engine.py"


def _tree():
    return ast.parse(_ENGINE_PATH.read_text())


def _step_function(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "step"
                ):
                    return item
    raise AssertionError("LLMEngine.step was not found")


def _attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def test_engine_imports_speculative_partition_builder():
    tree = _tree()

    imported = {
        alias.name
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            == "tinyvllm.engine.speculative_execution"
        )
        for alias in node.names
    }

    assert "build_engine_speculative_partition" in imported


def test_step_consumes_current_scheduler_selection_record():
    step = _step_function(_tree())
    calls = [
        node
        for node in ast.walk(step)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "build_engine_speculative_partition"
        )
    ]

    assert len(calls) == 1
    call = calls[0]
    paths = {
        _attribute_path(node)
        for node in ast.walk(call)
        if isinstance(node, ast.Attribute)
    }
    assert (
        "self",
        "scheduler",
        "last_speculative_selection",
    ) in paths
    assert (
        "self",
        "scheduler",
        "schedule_generation",
    ) in paths


def test_selected_row_guard_precedes_model_runner_execution():
    step = _step_function(_tree())
    guard_lines = []
    model_run_lines = []
    for node in ast.walk(step):
        if isinstance(node, ast.If):
            paths = {
                _attribute_path(item)
                for item in ast.walk(node.test)
                if isinstance(item, ast.Attribute)
            }
            if ("partition", "selected_sequences") in paths:
                if any(
                    isinstance(item, ast.Raise)
                    for item in node.body
                ):
                    guard_lines.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and _attribute_path(node.func)
            == ("self", "model_runner", "call")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "run"
        ):
            model_run_lines.append(node.lineno)

    assert len(guard_lines) == 2
    assert model_run_lines
    assert max(guard_lines) < min(model_run_lines)


def test_step_observation_exposes_speculative_selection():
    step = _step_function(_tree())
    observation_keys = set()
    for node in ast.walk(step):
        if (
            isinstance(node, ast.Assign)
            and any(
                _attribute_path(target)
                == ("self", "last_step_observation")
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            observation_keys.update(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            )

    assert {
        "speculative_schedule_generation",
        "speculative_selected_seq_ids",
        "speculative_suppressed_seq_ids",
    } <= observation_keys
