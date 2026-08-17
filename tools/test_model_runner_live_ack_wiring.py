from __future__ import annotations

import ast
from itertools import count
import os
import pickle
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
    ModelRunnerCommandAckCollector,
    ModelRunnerCommandEnvelope,
    execute_acknowledged_command,
)


def _load_class_method(relative_path, class_name, method_name, namespace):
    path = os.path.join(ROOT, relative_path)
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    compiled = compile(
        ast.fix_missing_locations(module),
        path,
        "exec",
    )
    exec(compiled, namespace)
    return namespace[method_name]


def _model_runner_method(name):
    namespace = {
        "count": count,
        "pickle": pickle,
        "ModelRunnerCommandEnvelope": ModelRunnerCommandEnvelope,
        "execute_acknowledged_command": execute_acknowledged_command,
    }
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
        namespace,
    )


def _engine_method(name):
    namespace = {
        "ModelRunnerCommandAckCollector": (
            ModelRunnerCommandAckCollector
        ),
    }
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        name,
        namespace,
    )


class _Buffer:
    def __init__(self, size=4096):
        self.buf = bytearray(size)


class _Event:
    def __init__(self):
        self.set_calls = 0
        self.clear_calls = 0
        self.wait_calls = 0

    def set(self):
        self.set_calls += 1

    def clear(self):
        self.clear_calls += 1

    def wait(self):
        self.wait_calls += 1


class _Sender:
    def __init__(self):
        self.values = []
        self.close_calls = 0

    def send(self, value):
        self.values.append(value)

    def close(self):
        self.close_calls += 1


class _Receiver:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _Process:
    def __init__(self, *, alive=True):
        self.alive = alive
        self.started = False
        self.join_calls = 0

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive

    def join(self):
        self.join_calls += 1


class _Context:
    def __init__(self):
        self.pipes = []

    def Pipe(self, duplex=False):
        assert duplex is False
        receiver = _Receiver()
        sender = _Sender()
        self.pipes.append((receiver, sender))
        return receiver, sender


def test_model_runner_call_preserves_fire_and_forget_local_result():
    call = _model_runner_method("call")
    captured = []

    class Runner:
        rank = 0

        def dispatch_command(
            self,
            method_name,
            *args,
            requires_ack,
        ):
            captured.append((method_name, args, requires_ack))

        def add(self, left, right):
            return left + right

    runner = Runner()

    assert call(runner, "add", 2, 5) == 7
    assert captured == [("add", (2, 5), False)]


def test_model_runner_dispatch_emits_monotonic_envelopes():
    dispatch = _model_runner_method("dispatch_command")
    written = []
    runner = types.SimpleNamespace(
        rank=0,
        world_size=2,
        _command_ids=count(),
        write_shm=written.append,
    )

    first = dispatch(
        runner,
        "prepare",
        11,
        requires_ack=True,
    )
    second = dispatch(
        runner,
        "run",
        22,
        requires_ack=False,
    )

    assert first == ModelRunnerCommandEnvelope(
        command_id=0,
        method_name="prepare",
        args=(11,),
        requires_ack=True,
    )
    assert second.command_id == 1
    assert written == [first, second]


def test_model_runner_shared_memory_envelope_and_legacy_decode():
    write_shm = _model_runner_method("write_shm")
    read_shm = _model_runner_method("read_shm")
    event = _Event()
    shared = _Buffer()
    rank_zero = types.SimpleNamespace(
        world_size=2,
        rank=0,
        shm=shared,
        event=[event],
    )
    envelope = ModelRunnerCommandEnvelope(
        command_id=3,
        method_name="prepare",
        args=(1, 2),
        requires_ack=True,
    )

    write_shm(rank_zero, envelope)

    worker = types.SimpleNamespace(
        world_size=2,
        rank=1,
        shm=shared,
        event=event,
        _command_ids=count(100),
    )
    assert read_shm(worker) == envelope
    assert event.set_calls == 1
    assert event.wait_calls == 1
    assert event.clear_calls == 1

    legacy = pickle.dumps(["run", 7, 8])
    shared.buf[0:4] = len(legacy).to_bytes(4, "little")
    shared.buf[4:4 + len(legacy)] = legacy
    converted = read_shm(worker)
    assert converted.method_name == "run"
    assert converted.args == (7, 8)
    assert converted.requires_ack is False
    assert converted.command_id == 100


def test_model_runner_worker_loop_ack_no_ack_and_exit():
    loop = _model_runner_method("loop")
    sender = _Sender()
    envelopes = iter((
        ModelRunnerCommandEnvelope(
            command_id=1,
            method_name="add",
            args=(2, 3),
            requires_ack=True,
        ),
        ModelRunnerCommandEnvelope(
            command_id=2,
            method_name="touch",
            args=(),
            requires_ack=False,
        ),
        ModelRunnerCommandEnvelope(
            command_id=3,
            method_name="exit",
            args=(),
            requires_ack=False,
        ),
    ))

    class Runner:
        rank = 1
        ack_sender = sender

        def read_shm(self):
            return next(envelopes)

        def add(self, left, right):
            return left + right

        def touch(self):
            self.touched = True

        def exit(self):
            self.exited = True

    runner = Runner()
    runner.touched = False
    runner.exited = False

    loop(runner)

    assert sender.values == [
        ModelRunnerCommandAck(
            command_id=1,
            rank=1,
            status="ok",
            result=5,
        )
    ]
    assert runner.touched is True
    assert runner.exited is True


def test_model_runner_constructor_has_ack_sender_contract():
    source = open(
        os.path.join(ROOT, "tinyvllm/engine/model_runner.py"),
        encoding="utf-8",
    ).read()
    assert "ack_sender=None" in source
    assert "self.ack_sender = ack_sender" in source
    assert "self._command_ids = count()" in source


def test_engine_tp1_acknowledged_call_is_local_only():
    call_ack = _engine_method("call_model_runner_acknowledged")

    class Runner:
        world_size = 1

        def add(self, left, right):
            return left + right

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=None,
        ps=[],
    )

    assert call_ack(
        engine,
        "add",
        4,
        6,
        timeout_s=1.0,
    ) == (10, ())


def test_engine_tp_acknowledged_call_collects_workers():
    call_ack = _engine_method("call_model_runner_acknowledged")
    rank_alive = _engine_method("_is_worker_rank_alive")
    envelope = ModelRunnerCommandEnvelope(
        command_id=8,
        method_name="prepare",
        args=(17,),
        requires_ack=True,
    )
    collector_calls = []

    class Runner:
        world_size = 3

        def dispatch_command(
            self,
            method_name,
            *args,
            requires_ack,
        ):
            assert (method_name, args, requires_ack) == (
                "prepare",
                (17,),
                True,
            )
            return envelope

        def prepare(self, value):
            return {"rank": 0, "value": value}

    worker_acks = (
        ModelRunnerCommandAck(
            command_id=8,
            rank=1,
            status="ok",
            result={"rank": 1},
        ),
        ModelRunnerCommandAck(
            command_id=8,
            rank=2,
            status="ok",
            result={"rank": 2},
        ),
    )

    class Collector:
        def collect(self, command_id, **kwargs):
            collector_calls.append((command_id, kwargs))
            return worker_acks

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=Collector(),
        ps=[
            _Process(alive=True),
            _Process(alive=True),
        ],
    )
    engine._is_worker_rank_alive = lambda rank: rank_alive(
        engine,
        rank,
    )

    result = call_ack(
        engine,
        "prepare",
        17,
        timeout_s=2.5,
    )

    assert result == ({"rank": 0, "value": 17}, worker_acks)
    assert collector_calls[0][0] == 8
    assert collector_calls[0][1]["expected_ranks"] == (1, 2)
    assert collector_calls[0][1]["timeout_s"] == 2.5
    assert collector_calls[0][1]["is_rank_alive"](1) is True


def test_engine_step_logits_authority_is_all_rank_and_rank_zero_only():
    enable = _engine_method(
        "enable_step_logits_authority_recording"
    )
    read = _engine_method("read_step_logits_authority")
    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return Tensor(("clone", self.value))

    class Runner:
        world_size = 4
        rank = 0

        def last_step_logits(self):
            calls.append(("read",))
            return Tensor("rank-zero-logits")

    worker_acks = tuple(
        types.SimpleNamespace(
            rank=rank,
            result={"rank": rank, "enabled": True},
        )
        for rank in range(1, 4)
    )

    def call_ack(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        assert method_name == "enable_step_logits_recording"
        return {"rank": 0, "enabled": args[0]}, worker_acks

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        call_model_runner_acknowledged=call_ack,
    )
    assert enable(engine, True, timeout_s=9.0) == {
        "enabled": True,
        "rank_inventory": [0, 1, 2, 3],
    }
    logits = read(engine)
    assert logits.value == ("clone", "rank-zero-logits")

    bad_engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            world_size=4,
            rank=1,
            last_step_logits=lambda: Tensor("bad"),
        ),
    )
    try:
        read(bad_engine)
    except RuntimeError as error:
        assert "rank zero" in str(error)
    else:
        raise AssertionError("non-root logits authority was accepted")


def test_engine_step_logits_authority_rejects_rank_ack_mismatch():
    enable = _engine_method(
        "enable_step_logits_authority_recording"
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {"rank": 0, "enabled": True},
            (
                types.SimpleNamespace(
                    rank=1,
                    result={"rank": 1, "enabled": True},
                ),
                types.SimpleNamespace(
                    rank=3,
                    result={"rank": 3, "enabled": True},
                ),
            ),
        ),
    )
    try:
        enable(engine, True, timeout_s=9.0)
    except ValueError as error:
        assert "rank inventory" in str(error)
    else:
        raise AssertionError("incomplete logits authority ack was accepted")


def test_model_runner_step_logits_recording_returns_ranked_ack():
    enable = _model_runner_method("enable_step_logits_recording")
    runner = types.SimpleNamespace(
        rank=3,
        _record_step_logits=False,
        _last_step_logits_cpu=object(),
    )
    assert enable(runner, True) == {
        "rank": 3,
        "enabled": True,
    }
    assert runner._record_step_logits is True
    assert runner._last_step_logits_cpu is None
    assert enable(runner, False) == {
        "rank": 3,
        "enabled": False,
    }
    assert runner._record_step_logits is False


def test_engine_local_exception_poisons_collector():
    call_ack = _engine_method("call_model_runner_acknowledged")
    poison_reasons = []

    class Runner:
        world_size = 2

        def dispatch_command(self, *args, **kwargs):
            return ModelRunnerCommandEnvelope(
                command_id=9,
                method_name="fail",
                args=(),
                requires_ack=True,
            )

        def fail(self):
            raise ValueError("rank zero failed")

    class Collector:
        def poison(self, reason):
            poison_reasons.append(reason)

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=Collector(),
        ps=[_Process()],
    )

    try:
        call_ack(
            engine,
            "fail",
            timeout_s=1.0,
        )
    except ValueError as error:
        assert str(error) == "rank zero failed"
    else:
        raise AssertionError("rank-zero exception was swallowed")
    assert "rank zero failed" in poison_reasons[0]


def test_engine_collector_failure_propagates():
    call_ack = _engine_method("call_model_runner_acknowledged")

    class Runner:
        world_size = 2

        def dispatch_command(self, *args, **kwargs):
            return ModelRunnerCommandEnvelope(
                command_id=19,
                method_name="prepare",
                args=(),
                requires_ack=True,
            )

        def prepare(self):
            return "local-ok"

    class Collector:
        def collect(self, *args, **kwargs):
            raise TimeoutError("worker ack timeout")

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=Collector(),
        ps=[_Process()],
        _is_worker_rank_alive=lambda rank: True,
    )

    try:
        call_ack(
            engine,
            "prepare",
            timeout_s=1.0,
        )
    except TimeoutError as error:
        assert str(error) == "worker ack timeout"
    else:
        raise AssertionError("collector failure was swallowed")


def test_engine_channel_helpers_create_close_and_map_liveness():
    create_channels = _engine_method(
        "_create_worker_ack_channels"
    )
    close_channels = _engine_method(
        "_close_worker_ack_channels"
    )
    rank_alive = _engine_method("_is_worker_rank_alive")
    context = _Context()

    receivers, senders = create_channels(context, 2)

    assert tuple(rank for rank, _ in receivers) == (1, 2)
    assert tuple(rank for rank, _ in senders) == (1, 2)
    engine = types.SimpleNamespace(
        ps=[
            _Process(alive=True),
            _Process(alive=False),
        ],
        model_runner_ack_receivers=receivers,
        model_runner_ack_parent_senders=senders,
    )
    assert rank_alive(engine, 1) is True
    assert rank_alive(engine, 2) is False
    close_channels(engine)
    assert all(
        receiver.close_calls == 1
        for _, receiver in receivers
    )
    assert all(sender.close_calls == 1 for _, sender in senders)


def test_engine_source_wires_pipe_sender_and_collector():
    source = open(
        os.path.join(ROOT, "tinyvllm/engine/llm_engine.py"),
        encoding="utf-8",
    ).read()
    assert "_create_worker_ack_channels" in source
    assert "ModelRunnerCommandAckCollector" in source
    assert "ack_sender" in source
    assert "sender.close()" in source
    tree = ast.parse(source)
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    step_node = next(
        method
        for method in engine_class.body
        if isinstance(method, ast.FunctionDef)
        and method.name == "step"
    )
    step_source = ast.get_source_segment(source, step_node)
    assert "call_model_runner_acknowledged" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "model runner live acknowledgement wiring tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
