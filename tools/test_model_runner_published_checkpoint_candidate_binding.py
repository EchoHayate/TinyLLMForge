from __future__ import annotations

import ast
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"
MODEL_FINGERPRINT = "a" * 64


class _FakeCandidate:

    def __init__(self, name="candidate-a"):
        self.name = name


class _FakeSlot:

    def __init__(self):
        self.candidate = None
        self.publish_calls = []

    def publish(self, candidate):
        self.publish_calls.append(candidate)
        if self.candidate is not None:
            raise RuntimeError("slot already occupied")
        self.candidate = candidate
        return candidate


def _load_runner_method(name):
    source = MODEL_RUNNER_PATH.read_text()
    tree = ast.parse(source, filename=str(MODEL_RUNNER_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _runner(rank=1):
    return types.SimpleNamespace(
        rank=rank,
        qwen35_loaded_checkpoint_candidate_slot=_FakeSlot(),
        bind_calls=[],
    )


def _successful_identity_row(rank):
    return {
        "participant_id": rank,
        "model_fingerprint": MODEL_FINGERPRINT,
        "layout_fingerprint": "layout-a",
        "dtype": "float32",
    }


def test_model_runner_publishes_candidate_only_to_local_slot():
    publish = _load_runner_method(
        "publish_qwen35_loaded_checkpoint_candidate"
    )
    runner = _runner()
    candidate = _FakeCandidate()

    result = publish(runner, candidate)

    assert result is candidate
    assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is (
        candidate
    )
    assert runner.qwen35_loaded_checkpoint_candidate_slot.publish_calls == [
        candidate
    ]


def test_model_runner_missing_published_candidate_returns_error_row():
    bind = _load_runner_method(
        "bind_published_qwen35_loaded_checkpoint_candidate"
    )
    runner = _runner(rank=2)

    row = bind(runner)

    assert row == {
        "participant_id": 2,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "error",
        "model_fingerprint": "",
        "layout_fingerprint": "",
        "dtype": "",
        "detail": "loaded checkpoint candidate is not published",
    }


def test_model_runner_binds_published_candidate_and_exact_repeat():
    bind = _load_runner_method(
        "bind_published_qwen35_loaded_checkpoint_candidate"
    )
    runner = _runner(rank=1)
    candidate = _FakeCandidate()
    runner.qwen35_loaded_checkpoint_candidate_slot.candidate = candidate

    def bind_candidate(value):
        runner.bind_calls.append(value)
        return _successful_identity_row(runner.rank)

    runner.bind_qwen35_loaded_checkpoint_candidate = bind_candidate

    first = bind(runner)
    second = bind(runner)

    expected = {
        "participant_id": 1,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound",
        "model_fingerprint": MODEL_FINGERPRINT,
        "layout_fingerprint": "layout-a",
        "dtype": "float32",
        "detail": "",
    }
    assert first == second == expected
    assert runner.bind_calls == [candidate, candidate]


def test_model_runner_candidate_binding_error_becomes_retryable_row():
    bind = _load_runner_method(
        "bind_published_qwen35_loaded_checkpoint_candidate"
    )
    runner = _runner(rank=3)
    candidate = _FakeCandidate()
    runner.qwen35_loaded_checkpoint_candidate_slot.candidate = candidate
    failures = [RuntimeError("injected candidate conflict")]

    def bind_candidate(value):
        runner.bind_calls.append(value)
        if failures:
            raise failures.pop()
        return _successful_identity_row(runner.rank)

    runner.bind_qwen35_loaded_checkpoint_candidate = bind_candidate

    first = bind(runner)
    second = bind(runner)

    assert first == {
        "participant_id": 3,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "error",
        "model_fingerprint": "",
        "layout_fingerprint": "",
        "dtype": "",
        "detail": "RuntimeError: injected candidate conflict",
    }
    assert second["status"] == "bound"
    assert runner.bind_calls == [candidate, candidate]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "model runner published checkpoint candidate binding tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
