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
class FakeAck:
    ticket_id: int
    participant_id: int
    operation: str
    status: str
    detail: str = ""


class Qwen35HybridPrefixPublicationParticipant:
    def __init__(self, participant_id, pool):
        self.participant_id = participant_id
        self.pool = pool
        self.calls = []
        self.results = {
            "prepare": FakeAck(
                ticket_id=7,
                participant_id=participant_id,
                operation="prepare",
                status="prepared",
            ),
            "precommit": FakeAck(
                ticket_id=7,
                participant_id=participant_id,
                operation="precommit",
                status="precommitted",
            ),
            "commit": FakeAck(
                ticket_id=7,
                participant_id=participant_id,
                operation="commit",
                status="finalized",
            ),
            "seal": FakeAck(
                ticket_id=7,
                participant_id=participant_id,
                operation="seal",
                status="committed",
            ),
            "rollback": FakeAck(
                ticket_id=7,
                participant_id=participant_id,
                operation="rollback",
                status="rolled_back",
            ),
        }

    def _call(self, operation, payload):
        self.calls.append((operation, payload))
        return self.results[operation]

    def prepare(self, payload):
        return self._call("prepare", payload)

    def precommit(self, payload):
        return self._call("precommit", payload)

    def commit(self, payload):
        return self._call("commit", payload)

    def seal(self, payload):
        return self._call("seal", payload)

    def rollback(self, payload):
        return self._call("rollback", payload)


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


def _runner_method(name, namespace=None):
    method_namespace = {
        "Qwen35HybridPrefixPublicationParticipant": (
            Qwen35HybridPrefixPublicationParticipant
        ),
    }
    if namespace is not None:
        method_namespace.update(namespace)
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
        method_namespace,
    )


def _bind_runner_helpers(runner):
    participant = _runner_method(
        "_qwen35_hybrid_prefix_publication_owner"
    )
    result = _runner_method(
        "_qwen35_hybrid_prefix_publication_result"
    )
    validate = _runner_method(
        "_validate_qwen35_hybrid_prefix_publication_ack"
    )
    payload = _runner_method(
        "_qwen35_hybrid_prefix_publication_payload"
    )
    runner._qwen35_hybrid_prefix_publication_owner = (
        lambda: participant(runner)
    )
    runner._qwen35_hybrid_prefix_publication_result = (
        lambda *args, **kwargs: result(*args, **kwargs)
    )
    runner._validate_qwen35_hybrid_prefix_publication_ack = (
        lambda *args, **kwargs: validate(runner, *args, **kwargs)
    )
    runner._qwen35_hybrid_prefix_publication_payload = (
        lambda payload_or_payloads: payload(
            runner,
            payload_or_payloads,
        )
    )
    return runner


class _Pool:
    pass


class _Bridge:
    def __init__(self, pool):
        self.pool = pool


class _Layout:
    fingerprint = "layout"
    bytes_per_slot = 64


class _OwnerPool(_Pool):
    capacity = 4
    layout = _Layout()


def test_model_runner_configuration_installs_one_owner_graph_atomically():
    pool = _OwnerPool()
    restore_participant = types.SimpleNamespace(
        participant_id=1,
        pool=pool,
    )
    publication_participant = (
        Qwen35HybridPrefixPublicationParticipant(1, pool)
    )
    owner = types.SimpleNamespace(
        pool=pool,
        participant=restore_participant,
        publication_participant=publication_participant,
        max_entries=8,
        max_bytes=4096,
    )
    builds = []

    def build_owner(
        candidate_pool,
        *,
        participant_id,
        max_entries,
        max_bytes,
    ):
        builds.append((
            candidate_pool,
            participant_id,
            max_entries,
            max_bytes,
        ))
        return owner

    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner",
        {
            "build_qwen35_hybrid_prefix_restore_owner": build_owner,
        },
    )
    runner = types.SimpleNamespace(
        rank=1,
        hybrid_state_runtime_bridge=_Bridge(pool),
        qwen35_hybrid_prefix_restore_owner=None,
        qwen35_hybrid_prefix_restore_participant=None,
        qwen35_hybrid_prefix_publication_participant=None,
    )
    runner.install_qwen35_hybrid_prefix_restore_participant = (
        lambda participant: setattr(
            runner,
            "qwen35_hybrid_prefix_restore_participant",
            participant,
        )
    )
    runner.install_qwen35_hybrid_prefix_publication_participant = (
        lambda participant: setattr(
            runner,
            "qwen35_hybrid_prefix_publication_participant",
            participant,
        )
    )

    first = configure(runner, 8, 4096)
    second = configure(runner, 8, 4096)

    assert first == second
    assert len(builds) == 1
    assert runner.qwen35_hybrid_prefix_restore_owner is owner
    assert (
        runner.qwen35_hybrid_prefix_restore_participant
        is restore_participant
    )
    assert (
        runner.qwen35_hybrid_prefix_publication_participant
        is publication_participant
    )

    blocked = types.SimpleNamespace(
        rank=1,
        hybrid_state_runtime_bridge=_Bridge(pool),
        qwen35_hybrid_prefix_restore_owner=None,
        qwen35_hybrid_prefix_restore_participant=None,
        qwen35_hybrid_prefix_publication_participant=(
            Qwen35HybridPrefixPublicationParticipant(1, pool)
        ),
    )
    blocked.install_qwen35_hybrid_prefix_restore_participant = (
        lambda participant: setattr(
            blocked,
            "qwen35_hybrid_prefix_restore_participant",
            participant,
        )
    )
    blocked.install_qwen35_hybrid_prefix_publication_participant = (
        lambda participant: setattr(
            blocked,
            "qwen35_hybrid_prefix_publication_participant",
            participant,
        )
    )
    try:
        configure(blocked, 8, 4096)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "configuration replaced a preinstalled participant"
        )
    assert blocked.qwen35_hybrid_prefix_restore_participant is None
    assert blocked.qwen35_hybrid_prefix_restore_owner is None


def test_model_runner_publication_install_validates_coherence():
    install = _runner_method(
        "install_qwen35_hybrid_prefix_publication_participant"
    )
    pool = _Pool()
    participant = Qwen35HybridPrefixPublicationParticipant(1, pool)
    runner = types.SimpleNamespace(
        rank=1,
        hybrid_state_runtime_bridge=_Bridge(pool),
        qwen35_hybrid_prefix_publication_participant=None,
    )

    install(runner, participant)
    install(runner, participant)

    assert (
        runner.qwen35_hybrid_prefix_publication_participant
        is participant
    )

    invalid_participants = (
        object(),
        Qwen35HybridPrefixPublicationParticipant(2, pool),
        Qwen35HybridPrefixPublicationParticipant(1, _Pool()),
    )
    for invalid in invalid_participants:
        fresh = types.SimpleNamespace(
            rank=1,
            hybrid_state_runtime_bridge=_Bridge(pool),
            qwen35_hybrid_prefix_publication_participant=None,
        )
        try:
            install(fresh, invalid)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(
                "invalid publication participant was installed"
            )

    replacement = Qwen35HybridPrefixPublicationParticipant(1, pool)
    try:
        install(runner, replacement)
    except RuntimeError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError(
            "publication participant replacement was accepted"
        )


def test_model_runner_publication_methods_delegate_and_fail_closed():
    methods = {
        "prepare": _runner_method(
            "prepare_hybrid_prefix_publication"
        ),
        "precommit": _runner_method(
            "precommit_hybrid_prefix_publication"
        ),
        "finalize": _runner_method(
            "finalize_hybrid_prefix_publication"
        ),
        "seal": _runner_method(
            "seal_hybrid_prefix_publication"
        ),
        "rollback": _runner_method(
            "rollback_hybrid_prefix_publication"
        ),
    }
    payload = FakePayload(ticket_id=7, participant_id=1)
    uninstalled = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_publication_participant=None,
    ))
    for operation, method in methods.items():
        try:
            method(uninstalled, payload)
        except RuntimeError as error:
            assert "not installed" in str(error)
        else:
            raise AssertionError(
                f"uninstalled publication {operation} was accepted"
            )

    participant = Qwen35HybridPrefixPublicationParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_publication_participant=participant,
    ))
    expected_statuses = {
        "prepare": "prepared",
        "precommit": "precommitted",
        "finalize": "finalized",
        "seal": "committed",
        "rollback": "rolled_back",
    }
    for operation, method in methods.items():
        assert method(runner, payload) == {
            "ticket_id": 7,
            "participant_id": 1,
            "operation": operation,
            "status": expected_statuses[operation],
            "detail": "",
        }
    assert participant.calls == [
        ("prepare", payload),
        ("precommit", payload),
        ("commit", payload),
        ("seal", payload),
        ("rollback", payload),
    ]


def test_model_runner_publication_methods_select_rank_payload():
    methods = {
        "prepare": _runner_method(
            "prepare_hybrid_prefix_publication"
        ),
        "precommit": _runner_method(
            "precommit_hybrid_prefix_publication"
        ),
        "finalize": _runner_method(
            "finalize_hybrid_prefix_publication"
        ),
        "seal": _runner_method(
            "seal_hybrid_prefix_publication"
        ),
        "rollback": _runner_method(
            "rollback_hybrid_prefix_publication"
        ),
    }
    payloads = (
        FakePayload(ticket_id=7, participant_id=0),
        FakePayload(ticket_id=7, participant_id=1),
    )
    participant = Qwen35HybridPrefixPublicationParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        rank=1,
        world_size=2,
        qwen35_hybrid_prefix_publication_participant=participant,
    ))

    for operation, method in methods.items():
        result = method(runner, payloads)
        assert result["participant_id"] == 1
        assert participant.calls[-1][1] is payloads[1]

    malformed = (
        payloads[:1],
        (payloads[0], payloads[0]),
        (
            payloads[0],
            FakePayload(ticket_id=7, participant_id=2),
        ),
    )
    calls_before = len(participant.calls)
    for payload_matrix in malformed:
        try:
            methods["prepare"](runner, payload_matrix)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "malformed publication payload matrix was accepted"
            )
    assert len(participant.calls) == calls_before


def test_model_runner_publication_methods_accept_exact_status_matrix():
    methods = {
        "prepare": _runner_method(
            "prepare_hybrid_prefix_publication"
        ),
        "precommit": _runner_method(
            "precommit_hybrid_prefix_publication"
        ),
        "finalize": _runner_method(
            "finalize_hybrid_prefix_publication"
        ),
        "seal": _runner_method(
            "seal_hybrid_prefix_publication"
        ),
        "rollback": _runner_method(
            "rollback_hybrid_prefix_publication"
        ),
    }
    participant = Qwen35HybridPrefixPublicationParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_publication_participant=participant,
    ))
    payload = FakePayload(ticket_id=7, participant_id=1)
    status_matrix = {
        "prepare": ("prepared", "rejected", "error"),
        "precommit": ("precommitted", "error"),
        "finalize": ("finalized", "error"),
        "seal": ("committed", "error"),
        "rollback": ("rolled_back", "error"),
    }
    for operation, statuses in status_matrix.items():
        participant_operation = (
            "commit" if operation == "finalize" else operation
        )
        for status in statuses:
            participant.results[participant_operation] = FakeAck(
                ticket_id=7,
                participant_id=1,
                operation=participant_operation,
                status=status,
                detail="detail",
            )
            result = methods[operation](runner, payload)
            assert result["status"] == status
            assert result["detail"] == "detail"


def test_model_runner_publication_methods_reject_malformed_ack():
    methods = {
        "prepare": _runner_method(
            "prepare_hybrid_prefix_publication"
        ),
        "precommit": _runner_method(
            "precommit_hybrid_prefix_publication"
        ),
        "finalize": _runner_method(
            "finalize_hybrid_prefix_publication"
        ),
        "seal": _runner_method(
            "seal_hybrid_prefix_publication"
        ),
        "rollback": _runner_method(
            "rollback_hybrid_prefix_publication"
        ),
    }
    participant = Qwen35HybridPrefixPublicationParticipant(1, _Pool())
    runner = _bind_runner_helpers(types.SimpleNamespace(
        qwen35_hybrid_prefix_publication_participant=participant,
    ))
    payload = FakePayload(ticket_id=7, participant_id=1)
    for operation, method in methods.items():
        participant_operation = (
            "commit" if operation == "finalize" else operation
        )
        invalid_acks = (
            FakeAck(8, 1, participant_operation, "error"),
            FakeAck(7, 2, participant_operation, "error"),
            FakeAck(7, 1, "wrong", "error"),
            FakeAck(7, 1, participant_operation, "unknown"),
            FakeAck(
                7,
                1,
                participant_operation,
                "error",
                detail=object(),
            ),
        )
        for acknowledgement in invalid_acks:
            participant.results[participant_operation] = acknowledgement
            try:
                method(runner, payload)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"malformed publication {operation} ack was accepted"
                )


def test_source_keeps_publication_disconnected_from_runtime_step():
    runner_source = open(
        os.path.join(ROOT, "tinyvllm/engine/model_runner.py"),
        encoding="utf-8",
    ).read()
    engine_source = open(
        os.path.join(ROOT, "tinyvllm/engine/llm_engine.py"),
        encoding="utf-8",
    ).read()
    assert (
        "qwen35_hybrid_prefix_publication_participant"
        in runner_source
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
        "model runner hybrid prefix publication method tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
