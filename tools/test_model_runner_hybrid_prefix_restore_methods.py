from __future__ import annotations

import ast
from dataclasses import dataclass
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        os.path.join(ROOT, *package_name.split("."))
    ]
    sys.modules[package_name] = package

from tinyvllm.engine.model_runner_command_ack import (
    ModelRunnerCommandAck,
)


@dataclass(frozen=True)
class FakePayload:
    ticket_id: int
    request_id: int = 17


@dataclass(frozen=True)
class FakePrepareAck:
    ticket_id: int
    participant_id: int
    status: str
    detail: str = ""


class Qwen35HybridPrefixRestoreParticipant:
    def __init__(self, participant_id, pool):
        self.participant_id = participant_id
        self.pool = pool
        self.calls = []
        self.prepare_result = FakePrepareAck(
            ticket_id=7,
            participant_id=participant_id,
            status="prepared",
        )

    def prepare(self, payload):
        self.calls.append(("prepare", payload))
        return self.prepare_result

    def validate_prepared(self, payload):
        self.calls.append(("validate", payload))

    def commit(self, payload):
        self.calls.append(("commit", payload))

    def rollback(self, payload):
        self.calls.append(("rollback", payload))


def _load_class_method(
    relative_path,
    class_name,
    method_name,
    namespace,
):
    path = os.path.join(ROOT, relative_path)
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source, filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    exec(
        compile(
            ast.fix_missing_locations(module),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def _runner_method(name):
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
        {
            "Qwen35HybridPrefixRestoreParticipant": (
                Qwen35HybridPrefixRestoreParticipant
            ),
        },
    )


def _bind_runner_helpers(runner):
    owner = _runner_method(
        "_qwen35_hybrid_prefix_restore_owner"
    )
    result = _runner_method(
        "_qwen35_hybrid_prefix_restore_result"
    )
    runner._qwen35_hybrid_prefix_restore_owner = (
        lambda: owner(runner)
    )
    runner._qwen35_hybrid_prefix_restore_result = (
        lambda *args, **kwargs: result(*args, **kwargs)
    )
    return runner


def _engine_method(name):
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        name,
        {},
    )


def _bind_engine_helpers(engine):
    poison = _engine_method("_poison_model_runner_ack_collector")
    validate = _engine_method(
        "_validate_hybrid_prefix_restore_results"
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    engine._validate_hybrid_prefix_restore_results = (
        lambda *args, **kwargs: validate(
            engine,
            *args,
            **kwargs,
        )
    )
    return engine


class _Pool:
    pass


class _Bridge:
    def __init__(self, pool):
        self.pool = pool


class _Collector:
    def __init__(self):
        self.poison_reasons = []

    def poison(self, reason):
        self.poison_reasons.append(reason)


def test_model_runner_install_validates_rank_type_and_pool_coherence():
    install = _runner_method(
        "install_qwen35_hybrid_prefix_restore_participant"
    )
    pool = _Pool()
    participant = Qwen35HybridPrefixRestoreParticipant(1, pool)
    runner = types.SimpleNamespace(
        rank=1,
        hybrid_state_runtime_bridge=_Bridge(pool),
        qwen35_hybrid_prefix_restore_participant=None,
    )

    install(runner, participant)
    install(runner, participant)

    assert runner.qwen35_hybrid_prefix_restore_participant is participant

    invalid_participants = (
        object(),
        Qwen35HybridPrefixRestoreParticipant(2, pool),
        Qwen35HybridPrefixRestoreParticipant(1, _Pool()),
    )
    for invalid in invalid_participants:
        fresh = types.SimpleNamespace(
            rank=1,
            hybrid_state_runtime_bridge=_Bridge(pool),
            qwen35_hybrid_prefix_restore_participant=None,
        )
        try:
            install(fresh, invalid)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid participant was installed")

    replacement = Qwen35HybridPrefixRestoreParticipant(1, pool)
    try:
        install(runner, replacement)
    except RuntimeError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError("participant replacement was accepted")


def test_model_runner_restore_methods_delegate_and_fail_closed():
    methods = {
        operation: _runner_method(
            f"{operation}_hybrid_prefix_restore"
        )
        for operation in (
            "prepare",
            "validate",
            "commit",
            "rollback",
        )
    }
    payload = FakePayload(ticket_id=7)
    uninstalled = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_restore_participant=None,
    ))
    for operation, method in methods.items():
        try:
            method(uninstalled, payload)
        except RuntimeError as error:
            assert "not installed" in str(error)
        else:
            raise AssertionError(
                f"uninstalled {operation} was accepted"
            )

    participant = Qwen35HybridPrefixRestoreParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_restore_participant=participant,
    ))

    prepared = methods["prepare"](runner, payload)
    assert prepared == {
        "ticket_id": 7,
        "participant_id": 1,
        "operation": "prepare",
        "status": "prepared",
        "detail": "",
    }

    participant.prepare_result = FakePrepareAck(
        ticket_id=7,
        participant_id=1,
        status="miss",
        detail="cache miss",
    )
    assert methods["prepare"](runner, payload)["status"] == "miss"
    participant.prepare_result = FakePrepareAck(
        ticket_id=7,
        participant_id=1,
        status="error",
        detail="restore failed",
    )
    assert methods["prepare"](runner, payload)["status"] == "error"

    for operation in ("validate", "commit", "rollback"):
        assert methods[operation](runner, payload) == {
            "ticket_id": 7,
            "participant_id": 1,
            "operation": operation,
            "status": "ok",
            "detail": "",
        }


def test_model_runner_prepare_rejects_participant_ack_identity_mismatch():
    prepare = _runner_method("prepare_hybrid_prefix_restore")
    participant = Qwen35HybridPrefixRestoreParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        rank=1,
        qwen35_hybrid_prefix_restore_participant=participant,
    ))
    payload = FakePayload(ticket_id=7)
    invalid_acks = (
        FakePrepareAck(
            ticket_id=8,
            participant_id=1,
            status="prepared",
        ),
        FakePrepareAck(
            ticket_id=7,
            participant_id=2,
            status="prepared",
        ),
        FakePrepareAck(
            ticket_id=7,
            participant_id=1,
            status="unknown",
        ),
    )
    for acknowledgement in invalid_acks:
        participant.prepare_result = acknowledgement
        try:
            prepare(runner, payload)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "participant acknowledgement identity mismatch was accepted"
            )


def test_engine_prepare_aggregation_orders_and_preserves_inner_status():
    prepare = _engine_method(
        "prepare_model_runner_hybrid_prefix_restore"
    )
    payload = FakePayload(ticket_id=9)
    worker_acks = (
        ModelRunnerCommandAck(
            command_id=1,
            rank=1,
            status="ok",
            result={
                "ticket_id": 9,
                "participant_id": 1,
                "operation": "prepare",
                "status": "miss",
                "detail": "",
            },
        ),
        ModelRunnerCommandAck(
            command_id=1,
            rank=2,
            status="ok",
            result={
                "ticket_id": 9,
                "participant_id": 2,
                "operation": "prepare",
                "status": "prepared",
                "detail": "",
            },
        ),
    )
    collector = _Collector()
    engine = _bind_engine_helpers(types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=3),
        model_runner_ack_collector=collector,
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {
                "ticket_id": 9,
                "participant_id": 0,
                "operation": "prepare",
                "status": "error",
                "detail": "local restore error",
            },
            worker_acks,
        ),
    ))

    results = prepare(engine, payload, timeout_s=2.0)

    assert tuple(row["participant_id"] for row in results) == (0, 1, 2)
    assert tuple(row["status"] for row in results) == (
        "error",
        "miss",
        "prepared",
    )
    assert collector.poison_reasons == []


def test_engine_malformed_nested_result_poisons():
    prepare = _engine_method(
        "prepare_model_runner_hybrid_prefix_restore"
    )
    payload = FakePayload(ticket_id=10)
    invalid_rows = (
        {"ticket_id": 11, "participant_id": 0,
         "operation": "prepare", "status": "prepared", "detail": ""},
        {"ticket_id": 10, "participant_id": 9,
         "operation": "prepare", "status": "prepared", "detail": ""},
        {"ticket_id": 10, "participant_id": 0,
         "operation": "commit", "status": "prepared", "detail": ""},
        {"ticket_id": 10, "participant_id": 0,
         "operation": "prepare", "status": "unknown", "detail": ""},
        "not-a-dict",
    )
    for invalid in invalid_rows:
        collector = _Collector()
        engine = _bind_engine_helpers(types.SimpleNamespace(
            model_runner=types.SimpleNamespace(world_size=1),
            model_runner_ack_collector=collector,
            call_model_runner_acknowledged=(
                lambda *args, invalid=invalid, **kwargs: (invalid, ())
            ),
        ))
        try:
            prepare(engine, payload, timeout_s=1.0)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("malformed nested result was accepted")
        assert collector.poison_reasons


def test_engine_rejects_worker_inner_rank_swaps():
    prepare = _engine_method(
        "prepare_model_runner_hybrid_prefix_restore"
    )
    payload = FakePayload(ticket_id=13)
    collector = _Collector()
    worker_acks = (
        ModelRunnerCommandAck(
            command_id=3,
            rank=1,
            status="ok",
            result={
                "ticket_id": 13,
                "participant_id": 2,
                "operation": "prepare",
                "status": "prepared",
                "detail": "",
            },
        ),
        ModelRunnerCommandAck(
            command_id=3,
            rank=2,
            status="ok",
            result={
                "ticket_id": 13,
                "participant_id": 1,
                "operation": "prepare",
                "status": "prepared",
                "detail": "",
            },
        ),
    )
    engine = _bind_engine_helpers(types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=3),
        model_runner_ack_collector=collector,
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {
                "ticket_id": 13,
                "participant_id": 0,
                "operation": "prepare",
                "status": "prepared",
                "detail": "",
            },
            worker_acks,
        ),
    ))

    try:
        prepare(engine, payload, timeout_s=1.0)
    except ValueError as error:
        assert "outer rank" in str(error)
    else:
        raise AssertionError("swapped worker identities were accepted")
    assert collector.poison_reasons


def test_engine_validate_commit_rollback_require_all_ok():
    helper = _engine_method(
        "_call_model_runner_hybrid_prefix_restore_operation"
    )
    payload = FakePayload(ticket_id=12)

    for operation in ("validate", "commit", "rollback"):
        collector = _Collector()
        engine = _bind_engine_helpers(types.SimpleNamespace(
            model_runner=types.SimpleNamespace(world_size=2),
            model_runner_ack_collector=collector,
            call_model_runner_acknowledged=lambda *args, operation=operation,
            **kwargs: (
                {
                    "ticket_id": 12,
                    "participant_id": 0,
                    "operation": operation,
                    "status": "ok",
                    "detail": "",
                },
                (
                    ModelRunnerCommandAck(
                        command_id=2,
                        rank=1,
                        status="ok",
                        result={
                            "ticket_id": 12,
                            "participant_id": 1,
                            "operation": operation,
                            "status": "ok",
                            "detail": "",
                        },
                    ),
                ),
            ),
        ))
        rows = helper(
            engine,
            operation,
            payload,
            timeout_s=1.0,
        )
        assert tuple(row["status"] for row in rows) == ("ok", "ok")
        assert collector.poison_reasons == []

    collector = _Collector()
    engine = _bind_engine_helpers(types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=1),
        model_runner_ack_collector=collector,
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {
                "ticket_id": 12,
                "participant_id": 0,
                "operation": "commit",
                "status": "error",
                "detail": "commit failed",
            },
            (),
        ),
    ))
    try:
        helper(
            engine,
            "commit",
            payload,
            timeout_s=1.0,
        )
    except RuntimeError as error:
        assert "commit failed" in str(error)
    else:
        raise AssertionError("non-ok commit result was accepted")
    assert collector.poison_reasons


def test_source_keeps_scheduler_and_step_fail_closed():
    runner_source = open(
        os.path.join(ROOT, "tinyvllm/engine/model_runner.py"),
        encoding="utf-8",
    ).read()
    engine_source = open(
        os.path.join(ROOT, "tinyvllm/engine/llm_engine.py"),
        encoding="utf-8",
    ).read()
    scheduler_source = open(
        os.path.join(ROOT, "tinyvllm/engine/scheduler.py"),
        encoding="utf-8",
    ).read()
    assert "qwen35_hybrid_prefix_restore_participant" in runner_source
    assert "prepare_model_runner_hybrid_prefix_restore" in engine_source
    assert (
        "hybrid prefix reuse requires aligned state snapshot"
        in scheduler_source
    )
    tree = ast.parse(engine_source)
    engine_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    step_node = next(
        node for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    step_source = ast.get_source_segment(engine_source, step_node)
    assert "prepare_model_runner_hybrid_prefix_restore" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "model runner hybrid prefix restore method tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
