from __future__ import annotations

import ast
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"
MODEL_FINGERPRINT = "a" * 64


def _validate_model_fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "model_fingerprint must be a lowercase SHA256 hex digest"
        )
    return value


class _FakeIdentity:

    def __init__(self, model_fingerprint, layout_fingerprint, dtype):
        self.model_fingerprint = model_fingerprint
        self.layout_fingerprint = layout_fingerprint
        self.dtype = dtype


class _FakePublisher:
    pass


class _FakeRestoreCoordinator:

    def __init__(self, engine, *, timeout_s):
        self.engine = engine
        self.timeout_s = float(timeout_s)


class _FakePublicationCoordinator:

    def __init__(self, engine, *, timeout_s):
        engine.calls.append(("coordinator_create", float(timeout_s)))
        self.engine = engine
        self.timeout_s = float(timeout_s)


def _load_engine_method(name):
    source = ENGINE_PATH.read_text()
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    namespace = {
        "Qwen35HybridPrefixEnginePublicationCoordinator": (
            _FakePublicationCoordinator
        ),
        "Qwen35HybridPrefixEngineRestoreCoordinator": (
            _FakeRestoreCoordinator
        ),
        "Qwen35HybridPrefixRuntimeIdentity": _FakeIdentity,
        "validate_qwen35_model_fingerprint": (
            _validate_model_fingerprint
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class _FakeEngine:

    def __init__(self):
        self.calls = []
        self.qwen35_hybrid_prefix_restore_configuration = None
        self.qwen35_hybrid_prefix_engine_restore_coordinator = None
        self.qwen35_hybrid_prefix_runtime_identity_configuration = None
        self.qwen35_hybrid_prefix_runtime_identity = None
        self.qwen35_hybrid_prefix_engine_publication_coordinator = None
        self.qwen35_hybrid_prefix_source_publisher_configuration = None
        self.qwen35_hybrid_prefix_source_publisher = None
        self.qwen35_hybrid_prefix_publication_runtime_configuration = None
        self.qwen35_hybrid_prefix_publication_runtime_publisher = None
        self.coordinator_install_failures = 0
        self.publisher_install_failures = 0

    def configure_qwen35_hybrid_prefix_restore(
        self,
        *,
        max_entries,
        max_bytes,
        representation="exact_restore",
        timeout_s,
    ):
        configuration = (
            max_entries,
            max_bytes,
            representation,
            float(timeout_s),
        )
        self.calls.append(("restore", configuration))
        current = self.qwen35_hybrid_prefix_restore_configuration
        if current is not None and current != configuration:
            raise RuntimeError("restore conflict")
        if self.qwen35_hybrid_prefix_engine_restore_coordinator is None:
            self.qwen35_hybrid_prefix_engine_restore_coordinator = (
                _FakeRestoreCoordinator(
                    self,
                    timeout_s=timeout_s,
                )
            )
        self.qwen35_hybrid_prefix_restore_configuration = configuration
        return self.qwen35_hybrid_prefix_engine_restore_coordinator

    def configure_qwen35_hybrid_prefix_runtime_identity(
        self,
        *,
        model_fingerprint,
        timeout_s,
    ):
        configuration = (
            model_fingerprint,
            float(timeout_s),
        )
        self.calls.append(("identity", configuration))
        current = self.qwen35_hybrid_prefix_runtime_identity_configuration
        if current is not None and current != configuration:
            raise RuntimeError("identity conflict")
        if self.qwen35_hybrid_prefix_runtime_identity is None:
            self.qwen35_hybrid_prefix_runtime_identity = _FakeIdentity(
                model_fingerprint,
                "layout-a",
                "float32",
            )
        self.qwen35_hybrid_prefix_runtime_identity_configuration = (
            configuration
        )
        return self.qwen35_hybrid_prefix_runtime_identity

    def install_qwen35_hybrid_prefix_engine_publication_coordinator(
        self,
        coordinator,
    ):
        self.calls.append(("coordinator_install", coordinator))
        if self.coordinator_install_failures:
            self.coordinator_install_failures -= 1
            raise RuntimeError("injected coordinator failure")
        current = self.qwen35_hybrid_prefix_engine_publication_coordinator
        if current is not None and current is not coordinator:
            raise RuntimeError("coordinator conflict")
        self.qwen35_hybrid_prefix_engine_publication_coordinator = (
            coordinator
        )

    def install_configured_qwen35_hybrid_prefix_source_publisher(self):
        self.calls.append(("publisher_install",))
        if self.publisher_install_failures:
            self.publisher_install_failures -= 1
            raise RuntimeError("injected publisher failure")
        identity = self.qwen35_hybrid_prefix_runtime_identity
        configuration = (
            identity.model_fingerprint,
            identity.layout_fingerprint,
            identity.dtype,
        )
        current = self.qwen35_hybrid_prefix_source_publisher_configuration
        if current is not None and current != configuration:
            raise RuntimeError("publisher conflict")
        if self.qwen35_hybrid_prefix_source_publisher is None:
            self.qwen35_hybrid_prefix_source_publisher = _FakePublisher()
        self.qwen35_hybrid_prefix_source_publisher_configuration = (
            configuration
        )
        return self.qwen35_hybrid_prefix_source_publisher


def _configure(engine, **overrides):
    method = _load_engine_method(
        "configure_qwen35_hybrid_prefix_publication_runtime"
    )
    arguments = {
        "model_fingerprint": MODEL_FINGERPRINT,
        "max_entries": 8,
        "max_bytes": 4096,
        "timeout_s": 0.25,
    }
    arguments.update(overrides)
    return method(engine, **arguments)


def _expect_error(callback, message):
    try:
        callback()
    except (ValueError, RuntimeError) as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_complete_argument_validation_precedes_child_calls():
    cases = (
        ({"model_fingerprint": "A" * 64}, "model_fingerprint"),
        ({"max_entries": True}, "max_entries"),
        ({"max_entries": 0}, "max_entries"),
        ({"max_bytes": True}, "max_bytes"),
        ({"max_bytes": 0}, "max_bytes"),
        ({"timeout_s": True}, "timeout_s"),
        ({"timeout_s": 0}, "timeout_s"),
    )
    for overrides, message in cases:
        engine = _FakeEngine()
        _expect_error(
            lambda: _configure(engine, **overrides),
            message,
        )
        assert engine.calls == []


def test_success_executes_strict_order_and_stores_completion():
    engine = _FakeEngine()

    publisher = _configure(engine)

    assert publisher is engine.qwen35_hybrid_prefix_source_publisher
    assert (
        engine.qwen35_hybrid_prefix_publication_runtime_publisher
        is publisher
    )
    assert (
        engine.qwen35_hybrid_prefix_publication_runtime_configuration
        == (
            MODEL_FINGERPRINT,
            8,
            4096,
            "exact_restore",
            0.25,
        )
    )
    assert [call[0] for call in engine.calls] == [
        "restore",
        "identity",
        "coordinator_create",
        "coordinator_install",
        "publisher_install",
    ]


def test_exact_completed_repeat_has_zero_child_side_effects():
    engine = _FakeEngine()
    first = _configure(engine)
    engine.calls.clear()

    second = _configure(engine)

    assert second is first
    assert engine.calls == []


def test_incoherent_completed_state_fails_before_child_calls():
    engine = _FakeEngine()
    _configure(engine)
    engine.calls.clear()
    engine.qwen35_hybrid_prefix_publication_runtime_publisher = None

    _expect_error(
        lambda: _configure(engine),
        "completion state is incomplete",
    )

    assert engine.calls == []


def test_completed_child_conflict_fails_before_child_calls():
    engine = _FakeEngine()
    _configure(engine)
    engine.calls.clear()
    engine.qwen35_hybrid_prefix_restore_configuration = (
        8,
        8192,
        "exact_restore",
        0.25,
    )

    _expect_error(
        lambda: _configure(engine),
        "restore configuration",
    )

    assert engine.calls == []


def test_restore_configuration_without_coordinator_fails_before_calls():
    engine = _FakeEngine()
    engine.qwen35_hybrid_prefix_restore_configuration = (
        8,
        4096,
        "exact_restore",
        0.25,
    )

    _expect_error(
        lambda: _configure(engine),
        "restore state is incomplete",
    )

    assert engine.calls == []


def test_exact_retry_resumes_after_coordinator_install_failure():
    engine = _FakeEngine()
    engine.coordinator_install_failures = 1

    _expect_error(
        lambda: _configure(engine),
        "injected coordinator failure",
    )
    assert (
        engine.qwen35_hybrid_prefix_publication_runtime_configuration
        is None
    )
    assert engine.qwen35_hybrid_prefix_restore_configuration is not None
    assert (
        engine.qwen35_hybrid_prefix_runtime_identity_configuration
        is not None
    )
    assert (
        engine.qwen35_hybrid_prefix_engine_publication_coordinator
        is None
    )

    engine.calls.clear()
    publisher = _configure(engine)

    assert publisher is engine.qwen35_hybrid_prefix_source_publisher
    assert [call[0] for call in engine.calls] == [
        "restore",
        "identity",
        "coordinator_create",
        "coordinator_install",
        "publisher_install",
    ]


def test_exact_retry_resumes_after_publisher_install_failure():
    engine = _FakeEngine()
    engine.publisher_install_failures = 1

    _expect_error(
        lambda: _configure(engine),
        "injected publisher failure",
    )
    coordinator = (
        engine.qwen35_hybrid_prefix_engine_publication_coordinator
    )
    assert coordinator is not None
    assert (
        engine.qwen35_hybrid_prefix_publication_runtime_configuration
        is None
    )

    engine.calls.clear()
    publisher = _configure(engine)

    assert publisher is engine.qwen35_hybrid_prefix_source_publisher
    assert (
        engine.qwen35_hybrid_prefix_engine_publication_coordinator
        is coordinator
    )
    assert [call[0] for call in engine.calls] == [
        "restore",
        "identity",
        "publisher_install",
    ]


def test_conflicting_completed_retry_fails_before_child_calls():
    engine = _FakeEngine()
    publisher = _configure(engine)
    engine.calls.clear()

    _expect_error(
        lambda: _configure(engine, max_bytes=8192),
        "already configured",
    )

    assert engine.calls == []
    assert (
        engine.qwen35_hybrid_prefix_publication_runtime_publisher
        is publisher
    )


def test_conflicting_partial_retry_fails_before_child_calls():
    partial_states = (
        (
            "restore",
            lambda engine: setattr(
                engine,
                "qwen35_hybrid_prefix_restore_configuration",
                (8, 8192, "exact_restore", 0.25),
            ),
        ),
        (
            "runtime identity",
            lambda engine: setattr(
                engine,
                "qwen35_hybrid_prefix_runtime_identity_configuration",
                ("b" * 64, 0.25),
            ),
        ),
        (
            "publication coordinator",
            lambda engine: setattr(
                engine,
                "qwen35_hybrid_prefix_engine_publication_coordinator",
                _FakePublicationCoordinator(
                    types.SimpleNamespace(calls=[]),
                    timeout_s=0.25,
                ),
            ),
        ),
    )
    for message, install_conflict in partial_states:
        engine = _FakeEngine()
        install_conflict(engine)
        engine.calls.clear()

        _expect_error(lambda: _configure(engine), message)

        assert engine.calls == []


def test_incoherent_publisher_partial_state_fails_before_child_calls():
    engine = _FakeEngine()
    engine.qwen35_hybrid_prefix_source_publisher = _FakePublisher()
    engine.qwen35_hybrid_prefix_source_publisher_configuration = (
        MODEL_FINGERPRINT,
        "layout-a",
        "float32",
    )

    _expect_error(lambda: _configure(engine), "source publisher")

    assert engine.calls == []


def test_engine_step_remains_runtime_configuration_free():
    source = ENGINE_PATH.read_text()
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    step_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    step_source = ast.unparse(step_node)
    assert (
        "configure_qwen35_hybrid_prefix_publication_runtime"
        not in step_source
    )
    assert (
        "qwen35_hybrid_prefix_publication_runtime_configuration"
        not in step_source
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
        "engine resumable prefix publication runtime configuration "
        f"tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
