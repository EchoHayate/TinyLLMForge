from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package


publication_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication",
    "tinyvllm/engine/qwen35_hybrid_prefix_engine_publication.py",
)
Qwen35HybridPrefixEnginePublicationCoordinator = (
    publication_module.Qwen35HybridPrefixEnginePublicationCoordinator
)


@dataclass(frozen=True)
class FakePayload:
    ticket_id: int
    participant_id: int


def _payloads(ticket_id=7):
    return (
        FakePayload(ticket_id, 0),
        FakePayload(ticket_id, 1),
    )


class _Engine:
    def __init__(self):
        self.model_runner = types.SimpleNamespace(world_size=2)
        self.calls = []
        self.statuses = {
            "prepare": ("prepared", "prepared"),
            "precommit": ("precommitted", "precommitted"),
            "finalize": ("finalized", "finalized"),
            "seal": ("committed", "committed"),
            "rollback": ("rolled_back", "rolled_back"),
        }
        self.failures = {}
        self.poison_reasons = []

    def _validate_hybrid_prefix_publication_payloads(self, payloads):
        payloads = tuple(sorted(
            payloads,
            key=lambda payload: payload.participant_id,
        ))
        if (
            len(payloads) != self.model_runner.world_size
            or tuple(
                payload.participant_id for payload in payloads
            )
            != tuple(range(self.model_runner.world_size))
            or len({payload.ticket_id for payload in payloads}) != 1
        ):
            raise ValueError("invalid payload matrix")
        return payloads

    def _phase(self, operation, payloads, timeout_s):
        self.calls.append((operation, payloads, timeout_s))
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure
        return tuple(
            {
                "ticket_id": payload.ticket_id,
                "participant_id": payload.participant_id,
                "operation": operation,
                "status": self.statuses[operation][
                    payload.participant_id
                ],
                "detail": "",
            }
            for payload in payloads
        )

    def prepare_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._phase("prepare", payloads, timeout_s)

    def precommit_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._phase("precommit", payloads, timeout_s)

    def finalize_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._phase("finalize", payloads, timeout_s)

    def seal_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._phase("seal", payloads, timeout_s)

    def rollback_model_runner_hybrid_prefix_publication(
        self,
        payloads,
        *,
        timeout_s,
    ):
        return self._phase("rollback", payloads, timeout_s)

    def _poison_model_runner_ack_collector(self, reason):
        self.poison_reasons.append(reason)


def _coordinator():
    engine = _Engine()
    coordinator = Qwen35HybridPrefixEnginePublicationCoordinator(
        engine,
        timeout_s=0.25,
    )
    return engine, coordinator


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError, TimeoutError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_engine_publication_transaction_success():
    engine, coordinator = _coordinator()
    payloads = _payloads()

    assert coordinator.publish(payloads) is True
    assert [call[0] for call in engine.calls] == [
        "prepare",
        "precommit",
        "finalize",
        "seal",
    ]
    assert coordinator.last_transaction.state == "committed"
    assert engine.poison_reasons == []


def test_prepare_rejection_rolls_back_all_and_returns_false():
    engine, coordinator = _coordinator()
    engine.statuses["prepare"] = ("prepared", "rejected")

    assert coordinator.publish(_payloads()) is False
    assert [call[0] for call in engine.calls] == [
        "prepare",
        "rollback",
    ]
    assert coordinator.last_transaction.state == "rolled_back"
    assert engine.poison_reasons == []


def test_business_phase_errors_roll_back_and_allow_reuse():
    cases = (
        ("prepare", ("prepared", "error")),
        ("precommit", ("precommitted", "error")),
        ("finalize", ("finalized", "error")),
    )
    for operation, statuses in cases:
        engine, coordinator = _coordinator()
        engine.statuses[operation] = statuses
        _expect_error(
            lambda: coordinator.publish(_payloads()),
            f"publication {operation} failed",
        )
        assert engine.calls[-1][0] == "rollback"
        assert coordinator.last_transaction.state == "rolled_back"
        assert engine.poison_reasons == []

        engine.statuses[operation] = {
            "prepare": ("prepared", "prepared"),
            "precommit": ("precommitted", "precommitted"),
            "finalize": ("finalized", "finalized"),
        }[operation]
        assert coordinator.publish(_payloads(ticket_id=8)) is True


def test_rollback_failure_poisons_and_blocks_reuse():
    engine, coordinator = _coordinator()
    engine.statuses["prepare"] = ("prepared", "error")
    engine.statuses["rollback"] = ("rolled_back", "error")

    _expect_error(
        lambda: coordinator.publish(_payloads()),
        "publication rollback failed",
    )
    assert engine.poison_reasons
    _expect_error(
        lambda: coordinator.publish(_payloads(ticket_id=8)),
        "poisoned",
    )


def test_seal_failure_poisons_and_blocks_reuse():
    engine, coordinator = _coordinator()
    engine.statuses["seal"] = ("committed", "error")

    _expect_error(
        lambda: coordinator.publish(_payloads()),
        "publication seal failed",
    )
    assert engine.poison_reasons
    _expect_error(
        lambda: coordinator.publish(_payloads(ticket_id=8)),
        "poisoned",
    )


def test_transport_failure_poisons_and_blocks_reuse():
    engine, coordinator = _coordinator()
    engine.failures["precommit"] = TimeoutError("worker timeout")

    _expect_error(
        lambda: coordinator.publish(_payloads()),
        "worker timeout",
    )
    assert engine.poison_reasons
    assert [call[0] for call in engine.calls] == [
        "prepare",
        "precommit",
    ]
    _expect_error(
        lambda: coordinator.publish(_payloads(ticket_id=8)),
        "poisoned",
    )


def _load_engine_method(name, namespace):
    path = ROOT / "tinyvllm/engine/llm_engine.py"
    source = path.read_text()
    tree = compile(source, str(path), "exec", ast.PyCF_ONLY_AST)
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
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace[name]


import ast


def test_engine_installs_and_exposes_explicit_publication_coordinator():
    namespace = {
        "Qwen35HybridPrefixEnginePublicationCoordinator": (
            Qwen35HybridPrefixEnginePublicationCoordinator
        ),
    }
    install = _load_engine_method(
        "install_qwen35_hybrid_prefix_engine_publication_coordinator",
        namespace,
    )
    publish = _load_engine_method(
        "publish_qwen35_hybrid_prefix",
        namespace,
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
        qwen35_hybrid_prefix_engine_publication_coordinator=None,
    )
    coordinator = Qwen35HybridPrefixEnginePublicationCoordinator(
        engine,
        timeout_s=0.25,
    )

    install(engine, coordinator)
    install(engine, coordinator)
    assert (
        engine.qwen35_hybrid_prefix_engine_publication_coordinator
        is coordinator
    )
    coordinator.publish = lambda payloads: ("published", payloads)
    assert publish(engine, _payloads()) == (
        "published",
        _payloads(),
    )

    replacement = Qwen35HybridPrefixEnginePublicationCoordinator(
        engine,
        timeout_s=0.25,
    )
    _expect_error(
        lambda: install(engine, replacement),
        "already installed",
    )
    wrong_engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
    )
    wrong_coordinator = (
        Qwen35HybridPrefixEnginePublicationCoordinator(
            wrong_engine,
            timeout_s=0.25,
        )
    )
    fresh_engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
        qwen35_hybrid_prefix_engine_publication_coordinator=None,
    )
    _expect_error(
        lambda: install(fresh_engine, wrong_coordinator),
        "target this LLMEngine",
    )
    _expect_error(
        lambda: install(fresh_engine, object()),
        "must be a",
    )
    uninstalled = types.SimpleNamespace(
        qwen35_hybrid_prefix_engine_publication_coordinator=None,
    )
    _expect_error(
        lambda: publish(uninstalled, _payloads()),
        "not installed",
    )


def test_engine_step_remains_publication_free():
    path = ROOT / "tinyvllm/engine/llm_engine.py"
    source = path.read_text()
    tree = ast.parse(source)
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
    step_source = ast.get_source_segment(source, step_node)
    assert "publish_qwen35_hybrid_prefix" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "engine hybrid prefix publication transaction tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
