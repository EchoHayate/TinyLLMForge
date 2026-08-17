from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import os
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class FakeKey:
    tensor_parallel_size: int = 2


@dataclass(frozen=True)
class FakePayload:
    ticket_id: int
    participant_id: int
    request_id: int = 17
    key: FakeKey = FakeKey()
    token_ids: tuple[int, ...] = (1, 2, 3, 4)
    block_identities: tuple[tuple[int, int, int], ...] = (
        (9, 4, 201),
    )


@dataclass(frozen=True)
class FakeWorkerAck:
    rank: int
    result: object


class _Collector:
    def __init__(self):
        self.poison_reasons = []

    def poison(self, reason):
        self.poison_reasons.append(reason)


def _load_engine_method(name):
    path = os.path.join(ROOT, "tinyvllm/engine/llm_engine.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source, filename=path)
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
        "Qwen35HybridPrefixPublicationPayload": FakePayload,
    }
    exec(
        compile(
            ast.fix_missing_locations(module),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _bind_engine_helpers(engine):
    poison = _load_engine_method(
        "_poison_model_runner_ack_collector"
    )
    validate_payloads = _load_engine_method(
        "_validate_hybrid_prefix_publication_payloads"
    )
    validate_results = _load_engine_method(
        "_validate_hybrid_prefix_publication_results"
    )
    call_phase = _load_engine_method(
        "_call_model_runner_hybrid_prefix_publication_phase"
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    engine._validate_hybrid_prefix_publication_payloads = (
        lambda payloads: validate_payloads(engine, payloads)
    )
    engine._validate_hybrid_prefix_publication_results = (
        lambda *args, **kwargs: validate_results(
            engine,
            *args,
            **kwargs,
        )
    )
    engine._call_model_runner_hybrid_prefix_publication_phase = (
        lambda *args, **kwargs: call_phase(
            engine,
            *args,
            **kwargs,
        )
    )
    return engine


def _payloads():
    return (
        FakePayload(ticket_id=7, participant_id=0),
        FakePayload(ticket_id=7, participant_id=1),
    )


def _row(participant_id, operation, status, detail=""):
    return {
        "ticket_id": 7,
        "participant_id": participant_id,
        "operation": operation,
        "status": status,
        "detail": detail,
    }


def test_engine_publication_payload_validation_orders_exact_matrix():
    validate = _load_engine_method(
        "_validate_hybrid_prefix_publication_payloads"
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
    )
    payloads = _payloads()

    assert validate(engine, tuple(reversed(payloads))) == payloads

    invalid = (
        payloads[:1],
        (payloads[0], payloads[0]),
        (
            payloads[0],
            replace(payloads[1], participant_id=2),
        ),
        (
            payloads[0],
            replace(payloads[1], ticket_id=8),
        ),
        (
            payloads[0],
            replace(payloads[1], request_id=18),
        ),
        (
            payloads[0],
            replace(payloads[1], token_ids=(5, 6, 7, 8)),
        ),
        (
            payloads[0],
            replace(
                payloads[1],
                key=FakeKey(tensor_parallel_size=3),
            ),
        ),
        [payloads[0], payloads[1]],
    )
    for payload_matrix in invalid:
        try:
            validate(engine, payload_matrix)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid publication payload matrix was accepted"
            )


def test_engine_publication_result_validation_orders_and_poisons():
    validate = _load_engine_method(
        "_validate_hybrid_prefix_publication_results"
    )
    poison = _load_engine_method(
        "_poison_model_runner_ack_collector"
    )
    collector = _Collector()
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
        model_runner_ack_collector=collector,
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    validate_payloads = _load_engine_method(
        "_validate_hybrid_prefix_publication_payloads"
    )
    engine._validate_hybrid_prefix_publication_payloads = (
        lambda payloads: validate_payloads(engine, payloads)
    )
    payloads = _payloads()
    rows = validate(
        engine,
        payloads,
        "prepare",
        _row(0, "prepare", "error", "local"),
        (
            FakeWorkerAck(
                rank=1,
                result=_row(1, "prepare", "rejected"),
            ),
        ),
        allowed_statuses={"prepared", "rejected", "error"},
    )

    assert tuple(row["participant_id"] for row in rows) == (0, 1)
    assert tuple(row["status"] for row in rows) == (
        "error",
        "rejected",
    )
    assert collector.poison_reasons == []

    invalid_rows = (
        _row(0, "prepare", "unknown"),
        _row(0, "seal", "prepared"),
        {
            **_row(0, "prepare", "prepared"),
            "extra": True,
        },
        "not-a-dict",
    )
    for invalid in invalid_rows:
        collector = _Collector()
        engine.model_runner_ack_collector = collector
        try:
            validate(
                engine,
                payloads,
                "prepare",
                invalid,
                (
                    FakeWorkerAck(
                        rank=1,
                        result=_row(1, "prepare", "prepared"),
                    ),
                ),
                allowed_statuses={
                    "prepared",
                    "rejected",
                    "error",
                },
            )
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(
                "malformed publication result was accepted"
            )
        assert collector.poison_reasons

    collector = _Collector()
    engine.model_runner_ack_collector = collector
    try:
        validate(
            engine,
            payloads,
            "prepare",
            _row(0, "prepare", "prepared"),
            (
                FakeWorkerAck(
                    rank=1,
                    result=_row(0, "prepare", "prepared"),
                ),
            ),
            allowed_statuses={"prepared", "rejected", "error"},
        )
    except ValueError as error:
        assert "outer rank" in str(error)
    else:
        raise AssertionError("rank-swapped publication result was accepted")
    assert collector.poison_reasons


def test_engine_publication_phases_dispatch_exact_methods():
    method_names = {
        "prepare": "prepare_model_runner_hybrid_prefix_publication",
        "precommit": (
            "precommit_model_runner_hybrid_prefix_publication"
        ),
        "finalize": "finalize_model_runner_hybrid_prefix_publication",
        "seal": "seal_model_runner_hybrid_prefix_publication",
        "rollback": "rollback_model_runner_hybrid_prefix_publication",
    }
    runner_methods = {
        "prepare": "prepare_hybrid_prefix_publication",
        "precommit": "precommit_hybrid_prefix_publication",
        "finalize": "finalize_hybrid_prefix_publication",
        "seal": "seal_hybrid_prefix_publication",
        "rollback": "rollback_hybrid_prefix_publication",
    }
    statuses = {
        "prepare": ("error", "rejected"),
        "precommit": ("precommitted", "error"),
        "finalize": ("finalized", "error"),
        "seal": ("committed", "error"),
        "rollback": ("rolled_back", "error"),
    }
    payloads = _payloads()

    for operation, method_name in method_names.items():
        calls = []

        def call_ack(
            runner_method,
            payload_matrix,
            *,
            timeout_s,
        ):
            calls.append((
                runner_method,
                payload_matrix,
                timeout_s,
            ))
            local_status, worker_status = statuses[operation]
            return (
                _row(0, operation, local_status),
                (
                    FakeWorkerAck(
                        rank=1,
                        result=_row(
                            1,
                            operation,
                            worker_status,
                        ),
                    ),
                ),
            )

        engine = _bind_engine_helpers(types.SimpleNamespace(
            model_runner=types.SimpleNamespace(world_size=2),
            model_runner_ack_collector=_Collector(),
            call_model_runner_acknowledged=call_ack,
        ))
        method = _load_engine_method(method_name)
        rows = method(engine, tuple(reversed(payloads)), timeout_s=2.5)

        assert calls == [
            (runner_methods[operation], payloads, 2.5)
        ]
        assert tuple(row["participant_id"] for row in rows) == (0, 1)
        assert tuple(row["status"] for row in rows) == statuses[operation]
        assert (
            engine.model_runner_ack_collector.poison_reasons
            == []
        )


def test_engine_publication_rejects_bad_matrix_before_dispatch():
    prepare = _load_engine_method(
        "prepare_model_runner_hybrid_prefix_publication"
    )
    dispatches = []
    engine = _bind_engine_helpers(types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
        model_runner_ack_collector=_Collector(),
        call_model_runner_acknowledged=(
            lambda *args, **kwargs: dispatches.append((args, kwargs))
        ),
    ))

    try:
        prepare(engine, _payloads()[:1], timeout_s=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "invalid payload matrix reached publication dispatch"
        )
    assert dispatches == []


def test_engine_step_remains_publication_free():
    path = os.path.join(ROOT, "tinyvllm/engine/llm_engine.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source, filename=path)
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
    assert "hybrid_prefix_publication" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "engine hybrid prefix publication transport tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
