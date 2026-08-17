from __future__ import annotations

import ast
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"
MODEL_FINGERPRINT = "a" * 64
AUTHORIZATION_SHA256 = "b" * 64


class _FakeRequest:

    def __init__(
        self,
        checkpoint_dir="/approved/model",
        model_fingerprint=MODEL_FINGERPRINT,
        max_tensor_bytes=8 << 20,
        authorization_sha256=AUTHORIZATION_SHA256,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_fingerprint = model_fingerprint
        self.max_tensor_bytes = max_tensor_bytes
        self.authorization_sha256 = authorization_sha256

    def __eq__(self, other):
        return (
            type(other) is _FakeRequest
            and self.checkpoint_dir == other.checkpoint_dir
            and self.model_fingerprint == other.model_fingerprint
            and self.max_tensor_bytes == other.max_tensor_bytes
            and self.authorization_sha256
            == other.authorization_sha256
        )


class _FakeCandidate:

    def __init__(self, model_fingerprint=MODEL_FINGERPRINT):
        self.model_fingerprint = model_fingerprint


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


def _validate_request(value):
    if type(value) is not _FakeRequest:
        raise ValueError("request must be an exact request")
    return value


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
    namespace = {
        "Qwen35LoadedCheckpointCandidate": _FakeCandidate,
        "validate_qwen35_checkpoint_candidate_load_request": (
            _validate_request
        ),
    }
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
        qwen35_checkpoint_candidate_loader=None,
        qwen35_checkpoint_candidate_loader_authorization_sha256=None,
        qwen35_checkpoint_candidate_load_configuration=None,
        qwen35_checkpoint_candidate_load_request=None,
        qwen35_loaded_checkpoint_candidate_slot=_FakeSlot(),
    )


def _expect_error(callback, message):
    try:
        callback()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_loader_installation_is_exact_idempotent_and_conflict_safe():
    install = _load_runner_method(
        "install_qwen35_checkpoint_candidate_loader"
    )
    runner = _runner()
    loader = lambda request: request

    install(
        runner,
        loader,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    install(
        runner,
        loader,
        authorization_sha256=AUTHORIZATION_SHA256,
    )

    assert runner.qwen35_checkpoint_candidate_loader is loader
    assert (
        runner.qwen35_checkpoint_candidate_loader_authorization_sha256
        == AUTHORIZATION_SHA256
    )
    _expect_error(
        lambda: install(
            runner,
            lambda request: request,
            authorization_sha256=AUTHORIZATION_SHA256,
        ),
        "already installed",
    )
    _expect_error(
        lambda: install(
            _runner(),
            object(),
            authorization_sha256=AUTHORIZATION_SHA256,
        ),
        "callable",
    )
    _expect_error(
        lambda: install(
            _runner(),
            loader,
            authorization_sha256="bad",
        ),
        "authorization_sha256",
    )


def test_load_requires_coherent_installed_loader_and_authorization():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    request = _FakeRequest()
    calls = []
    cases = (
        (_runner(), "loader is not installed"),
        (
            types.SimpleNamespace(
                **{
                    **_runner().__dict__,
                    "qwen35_checkpoint_candidate_loader": (
                        lambda value: calls.append(value)
                    ),
                }
            ),
            "loader state is incomplete",
        ),
    )
    for runner, detail in cases:
        row = load(runner, request)
        assert row["status"] == "error"
        assert detail in row["detail"]
        assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is None
    assert calls == []

    runner = _runner()
    runner.qwen35_checkpoint_candidate_loader = (
        lambda value: calls.append(value)
    )
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        "c" * 64
    )
    row = load(runner, request)
    assert row["status"] == "error"
    assert "authorization" in row["detail"]
    assert calls == []


def test_loader_failure_or_invalid_candidate_leaves_state_pristine():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    request = _FakeRequest()
    outputs = (
        RuntimeError("injected load failure"),
        object(),
        _FakeCandidate("c" * 64),
    )
    messages = (
        "injected load failure",
        "exact Qwen35LoadedCheckpointCandidate",
        "fingerprint",
    )
    for output, message in zip(outputs, messages):
        runner = _runner()

        def loader(_request, output=output):
            if isinstance(output, Exception):
                raise output
            return output

        runner.qwen35_checkpoint_candidate_loader = loader
        runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
            AUTHORIZATION_SHA256
        )

        row = load(runner, request)

        assert row["status"] == "error"
        assert message in row["detail"]
        assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is None
        assert runner.qwen35_checkpoint_candidate_load_configuration is None
        assert runner.qwen35_checkpoint_candidate_load_request is None


def test_success_publishes_once_and_exact_repeat_skips_loader():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    runner = _runner(rank=2)
    request = _FakeRequest()
    candidate = _FakeCandidate()
    calls = []

    def loader(value):
        calls.append(value)
        return candidate

    runner.qwen35_checkpoint_candidate_loader = loader
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        AUTHORIZATION_SHA256
    )

    first = load(runner, request)
    second = load(runner, _FakeRequest())

    expected = {
        "participant_id": 2,
        "operation": "load_checkpoint_candidate",
        "status": "published",
        "model_fingerprint": MODEL_FINGERPRINT,
        "detail": "",
    }
    assert first == second == expected
    assert calls == [request]
    assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is (
        candidate
    )
    assert runner.qwen35_loaded_checkpoint_candidate_slot.publish_calls == [
        candidate
    ]
    assert runner.qwen35_checkpoint_candidate_load_configuration == (
        "/approved/model",
        MODEL_FINGERPRINT,
        8 << 20,
        AUTHORIZATION_SHA256,
    )
    assert runner.qwen35_checkpoint_candidate_load_request is request


def test_conflicting_repeat_fails_without_loader_or_replacement():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    runner = _runner()
    first_request = _FakeRequest()
    second_request = _FakeRequest(max_tensor_bytes=16 << 20)
    candidate = _FakeCandidate()
    calls = []

    def loader(value):
        calls.append(value)
        return candidate

    runner.qwen35_checkpoint_candidate_loader = loader
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        AUTHORIZATION_SHA256
    )
    assert load(runner, first_request)["status"] == "published"

    row = load(runner, second_request)

    assert row["status"] == "error"
    assert "already completed" in row["detail"]
    assert calls == [first_request]
    assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is (
        candidate
    )


def test_damaged_completed_loader_state_fails_without_reload():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    runner = _runner()
    request = _FakeRequest()
    candidate = _FakeCandidate()
    calls = []

    def loader(value):
        calls.append(value)
        return candidate

    runner.qwen35_checkpoint_candidate_loader = loader
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        AUTHORIZATION_SHA256
    )
    assert load(runner, request)["status"] == "published"

    runner.qwen35_checkpoint_candidate_loader = None
    row = load(runner, _FakeRequest())

    assert row["status"] == "error"
    assert "completion state is incomplete" in row["detail"]
    assert calls == [request]
    assert runner.qwen35_loaded_checkpoint_candidate_slot.candidate is (
        candidate
    )


def test_invalid_request_returns_bounded_error_without_loader_call():
    load = _load_runner_method(
        "load_and_publish_qwen35_checkpoint_candidate"
    )
    runner = _runner(rank=4)
    calls = []
    runner.qwen35_checkpoint_candidate_loader = (
        lambda value: calls.append(value)
    )
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        AUTHORIZATION_SHA256
    )

    row = load(runner, object())

    assert row["participant_id"] == 4
    assert row["status"] == "error"
    assert row["model_fingerprint"] == ""
    assert len(row["detail"].encode("utf-8")) <= 4096
    assert calls == []


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "model runner authorized checkpoint loader tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
