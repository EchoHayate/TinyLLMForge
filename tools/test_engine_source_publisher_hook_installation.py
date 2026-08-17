from __future__ import annotations

import ast
from pathlib import Path
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"


class _FakePublisher:

    def __init__(
        self,
        engine,
        *,
        model_fingerprint,
        layout_fingerprint,
        dtype,
    ):
        self.engine = engine
        self.model_fingerprint = model_fingerprint
        self.layout_fingerprint = layout_fingerprint
        self.dtype = dtype
        self.published = []

    def publish(self, sequence):
        self.published.append(sequence)
        return True


class _FakeIdentity:

    def __init__(
        self,
        *,
        model_fingerprint,
        layout_fingerprint,
        dtype,
    ):
        self.model_fingerprint = model_fingerprint
        self.layout_fingerprint = layout_fingerprint
        self.dtype = dtype


class _Scheduler:

    def __init__(self):
        self.installed_hooks = []
        self.failure = None

    def install_prefill_commit_hook(self, hook):
        if self.failure is not None:
            raise self.failure
        self.installed_hooks.append(hook)


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
        "Qwen35HybridPrefixSourcePublisher": _FakePublisher,
        "Qwen35HybridPrefixRuntimeIdentity": _FakeIdentity,
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


def _engine():
    return types.SimpleNamespace(
        scheduler=_Scheduler(),
        qwen35_hybrid_prefix_source_publisher=None,
        qwen35_hybrid_prefix_source_publisher_hook=None,
        qwen35_hybrid_prefix_source_publisher_configuration=None,
    )


def _install(engine, **overrides):
    install = _load_engine_method(
        "install_qwen35_hybrid_prefix_source_publisher"
    )
    arguments = {
        "model_fingerprint": "model-a",
        "layout_fingerprint": "layout-a",
        "dtype": "float32",
    }
    arguments.update(overrides)
    return install(engine, **arguments)


def test_engine_source_publisher_is_default_off():
    source = ENGINE_PATH.read_text()
    for assignment in (
        "self.qwen35_hybrid_prefix_source_publisher = None",
        "self.qwen35_hybrid_prefix_source_publisher_hook = None",
        "self.qwen35_hybrid_prefix_source_publisher_configuration = None",
    ):
        assert assignment in source


def test_engine_installs_exact_source_publisher_hook():
    engine = _engine()

    publisher = _install(engine)

    assert isinstance(publisher, _FakePublisher)
    assert publisher.engine is engine
    assert publisher.model_fingerprint == "model-a"
    assert publisher.layout_fingerprint == "layout-a"
    assert publisher.dtype == "float32"
    assert engine.qwen35_hybrid_prefix_source_publisher is publisher
    hook = engine.qwen35_hybrid_prefix_source_publisher_hook
    assert len(engine.scheduler.installed_hooks) == 1
    assert engine.scheduler.installed_hooks[0] is hook
    assert hook.__self__ is publisher
    assert hook.__func__ is _FakePublisher.publish
    assert engine.qwen35_hybrid_prefix_source_publisher_configuration == (
        "model-a",
        "layout-a",
        "float32",
    )


def test_engine_source_publisher_same_configuration_is_idempotent():
    engine = _engine()
    first = _install(engine)
    first_hook = engine.qwen35_hybrid_prefix_source_publisher_hook

    second = _install(engine)

    assert second is first
    assert engine.qwen35_hybrid_prefix_source_publisher_hook is first_hook
    assert engine.scheduler.installed_hooks == [first_hook]


def test_engine_source_publisher_replacement_fails_without_mutation():
    engine = _engine()
    publisher = _install(engine)
    hook = engine.qwen35_hybrid_prefix_source_publisher_hook
    configuration = (
        engine.qwen35_hybrid_prefix_source_publisher_configuration
    )

    try:
        _install(engine, layout_fingerprint="layout-b")
    except RuntimeError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError("source publisher replacement was accepted")

    assert engine.qwen35_hybrid_prefix_source_publisher is publisher
    assert engine.qwen35_hybrid_prefix_source_publisher_hook is hook
    assert (
        engine.qwen35_hybrid_prefix_source_publisher_configuration
        == configuration
    )
    assert engine.scheduler.installed_hooks == [hook]


def test_engine_source_publisher_installation_failure_is_atomic():
    engine = _engine()
    engine.scheduler.failure = RuntimeError("injected hook failure")

    try:
        _install(engine)
    except RuntimeError as error:
        assert str(error) == "injected hook failure"
    else:
        raise AssertionError("Scheduler hook failure was swallowed")

    assert engine.qwen35_hybrid_prefix_source_publisher is None
    assert engine.qwen35_hybrid_prefix_source_publisher_hook is None
    assert (
        engine.qwen35_hybrid_prefix_source_publisher_configuration
        is None
    )
    assert engine.scheduler.installed_hooks == []


def test_engine_step_does_not_install_or_call_source_publisher():
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
    assert "install_qwen35_hybrid_prefix_source_publisher" not in step_source
    assert "qwen35_hybrid_prefix_source_publisher" not in step_source


def test_configured_installer_requires_canonical_identity():
    install = _load_engine_method(
        "install_configured_qwen35_hybrid_prefix_source_publisher"
    )
    engine = _engine()
    engine.qwen35_hybrid_prefix_runtime_identity = None

    try:
        install(engine)
    except RuntimeError as error:
        assert "runtime identity" in str(error)
    else:
        raise AssertionError("missing runtime identity was accepted")
    assert engine.scheduler.installed_hooks == []


def test_configured_installer_delegates_exact_identity_and_is_idempotent():
    install = _load_engine_method(
        "install_configured_qwen35_hybrid_prefix_source_publisher"
    )
    explicit = _load_engine_method(
        "install_qwen35_hybrid_prefix_source_publisher"
    )
    engine = _engine()
    engine.qwen35_hybrid_prefix_runtime_identity = _FakeIdentity(
        model_fingerprint="a" * 64,
        layout_fingerprint="layout-a",
        dtype=torch.float32,
    )
    calls = []

    def install_explicit(**kwargs):
        calls.append(kwargs)
        return explicit(engine, **kwargs)

    engine.install_qwen35_hybrid_prefix_source_publisher = (
        install_explicit
    )

    first = install(engine)
    second = install(engine)

    assert second is first
    assert calls == [
        {
            "model_fingerprint": "a" * 64,
            "layout_fingerprint": "layout-a",
            "dtype": torch.float32,
        },
        {
            "model_fingerprint": "a" * 64,
            "layout_fingerprint": "layout-a",
            "dtype": torch.float32,
        },
    ]
    assert len(engine.scheduler.installed_hooks) == 1


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "engine source publisher hook installation tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
