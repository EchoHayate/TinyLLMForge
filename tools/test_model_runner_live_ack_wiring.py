from __future__ import annotations

import ast
import builtins
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
from tinyvllm.engine.model_runner_command_timeline import (
    CommandClockIdentity,
    CommandTraceIdentity,
    ModelRunnerCommandTimelineRecorder,
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
        "CommandTraceIdentity": CommandTraceIdentity,
        "read_command_clock_identity": lambda: CommandClockIdentity(
            boot_id="boot",
            implementation="clock_gettime(CLOCK_MONOTONIC)",
            resolution_s=1e-9,
            monotonic=True,
            adjustable=False,
            captured_at_unix_ns=1,
        ),
        "ModelRunnerCommandTimelineRecorder": (
            ModelRunnerCommandTimelineRecorder
        ),
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
    def __init__(self, on_set=None):
        self.set_calls = 0
        self.clear_calls = 0
        self.wait_calls = 0
        self.on_set = on_set

    def set(self):
        self.set_calls += 1
        if self.on_set is not None:
            self.on_set()

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


class _Timeline:
    def __init__(self, *, enabled):
        self.enabled = enabled
        self.dispatches = []
        self.receives = []
        self.ack_waits = []
        self.ack_wait_starts = {}

    def record_dispatch(self, identity):
        self.dispatches.append(identity)

    def record_worker_receive(
        self,
        identity,
        *,
        event_woken_monotonic_ns,
        envelope_read_monotonic_ns,
    ):
        self.receives.append((
            identity,
            event_woken_monotonic_ns,
            envelope_read_monotonic_ns,
        ))

    def record_ack_wait(self, command_id, *, started_ns, finished_ns):
        self.ack_waits.append((command_id, started_ns, finished_ns))

    def record_ack_wait_start(self, command_id, *, started_ns):
        self.ack_wait_starts[command_id] = started_ns

    def record_ack_wait_end(self, command_id, *, finished_ns):
        self.ack_waits.append((
            command_id,
            self.ack_wait_starts.pop(command_id),
            finished_ns,
        ))


def _step_trace():
    return types.SimpleNamespace(
        engine_step_id=7,
        repeat_index=3,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
    )


def _decode_shared_envelope(shared):
    n = int.from_bytes(shared.buf[0:4], "little")
    return pickle.loads(shared.buf[4:n + 4])


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
            return ModelRunnerCommandEnvelope(
                command_id=11,
                method_name=method_name,
                args=tuple(args),
                requires_ack=requires_ack,
            )

        def execute_command_envelope(self, envelope):
            assert envelope.command_id == 11
            return self.add(*envelope.args)

        @staticmethod
        def add(left, right):
            return left + right

    runner = Runner()

    assert call(runner, "add", 2, 5) == 7
    assert captured == [("add", (2, 5), False)]


def test_model_runner_dispatch_traces_only_enabled_active_repeat():
    dispatch = _model_runner_method("dispatch_command")
    written = []
    disabled = types.SimpleNamespace(
        rank=0,
        world_size=2,
        _command_ids=count(),
        command_timeline=_Timeline(enabled=False),
        _command_timeline_clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("disabled timeline read the clock")
        ),
        _active_command_timeline_trace=_step_trace,
        write_shm=written.append,
    )

    disabled_envelope = dispatch(
        disabled,
        "run",
        1,
        requires_ack=False,
    )
    assert disabled_envelope.trace_identity is None

    no_context = types.SimpleNamespace(
        rank=0,
        world_size=1,
        _command_ids=count(10),
        command_timeline=_Timeline(enabled=True),
        _command_timeline_clock_ns=iter((100, 200)).__next__,
        _active_command_timeline_trace=lambda: None,
        write_shm=written.append,
    )
    assert dispatch(
        no_context,
        "run",
        2,
        requires_ack=False,
    ).trace_identity is None

    timeline = _Timeline(enabled=True)
    traced = types.SimpleNamespace(
        rank=0,
        world_size=2,
        _command_ids=count(20),
        command_timeline=timeline,
        _command_timeline_clock_ns=iter((300, 400)).__next__,
        _active_command_timeline_trace=_step_trace,
        write_shm=written.append,
    )
    envelope = dispatch(
        traced,
        "run",
        3,
        requires_ack=True,
    )

    assert envelope.trace_identity == CommandTraceIdentity(
        command_id=20,
        method_name="run",
        requires_ack=True,
        engine_step_id=7,
        repeat_index=3,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
        dispatch_started_monotonic_ns=300,
        dispatch_published_monotonic_ns=400,
    )
    assert timeline.dispatches == [envelope.trace_identity]


def test_model_runner_management_dispatch_ignores_stale_measured_trace():
    dispatch = _model_runner_method("dispatch_command")
    execute = _model_runner_method("execute_command_envelope")
    snapshot = _model_runner_method("command_timeline_snapshot")
    timeline = ModelRunnerCommandTimelineRecorder(
        rank=0,
        max_rows=8,
        clock_identity=CommandClockIdentity(
            boot_id="boot",
            implementation="clock_gettime(CLOCK_MONOTONIC)",
            resolution_s=1e-9,
            monotonic=True,
            adjustable=False,
            captured_at_unix_ns=1,
        ),
    )
    runner = types.SimpleNamespace(
        rank=0,
        world_size=1,
        _command_ids=count(),
        command_timeline=timeline,
        _command_timeline_clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("management command read timeline clock")
        ),
        _active_command_timeline_trace=_step_trace,
        write_shm=lambda envelope: None,
    )
    runner.command_timeline_snapshot = types.MethodType(snapshot, runner)

    envelopes = tuple(
        dispatch(
            runner,
            method_name,
            *(True, 8) if method_name == "configure_command_timeline" else (),
            requires_ack=True,
        )
        for method_name in (
            "configure_command_timeline",
            "reset_command_timeline",
            "command_timeline_snapshot",
        )
    )

    assert all(envelope.trace_identity is None for envelope in envelopes)
    assert execute(runner, envelopes[-1])["rows"] == []


def test_model_runner_lazy_engine_step_import_only_suppresses_absent_module():
    read_trace = _model_runner_method("_read_active_engine_step_trace")
    original_import = builtins.__import__

    def import_with_missing_timeline(name, *args, **kwargs):
        if name == "tinyvllm.engine.engine_step_timeline":
            raise ModuleNotFoundError(
                "missing engine step timeline",
                name=name,
            )
        return original_import(name, *args, **kwargs)

    builtins.__import__ = import_with_missing_timeline
    try:
        assert read_trace() is None
    finally:
        builtins.__import__ = original_import

    def import_with_missing_dependency(name, *args, **kwargs):
        if name == "tinyvllm.engine.engine_step_timeline":
            raise ModuleNotFoundError(
                "missing nested dependency",
                name="timeline_nested_dependency",
            )
        return original_import(name, *args, **kwargs)

    builtins.__import__ = import_with_missing_dependency
    try:
        try:
            read_trace()
        except ModuleNotFoundError as error:
            assert error.name == "timeline_nested_dependency"
        else:
            raise AssertionError("nested timeline import failure was hidden")
    finally:
        builtins.__import__ = original_import


def test_model_runner_dispatch_emits_monotonic_envelopes():
    dispatch = _model_runner_method("dispatch_command")
    written = []
    runner = types.SimpleNamespace(
        rank=0,
        world_size=2,
        _command_ids=count(),
        command_timeline=_Timeline(enabled=False),
        _command_timeline_clock_ns=lambda: 0,
        _active_command_timeline_trace=lambda: None,
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
        command_timeline=_Timeline(enabled=False),
        _command_timeline_clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("disabled read_shm read the clock")
        ),
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


def test_model_runner_write_serializes_final_publish_before_event_set():
    write_shm = _model_runner_method("write_shm")
    shared = _Buffer()
    seen = []
    event = _Event(
        on_set=lambda: seen.append(_decode_shared_envelope(shared))
    )
    envelope = ModelRunnerCommandEnvelope(
        command_id=23,
        method_name="prepare",
        args=(),
        requires_ack=True,
        trace_identity=CommandTraceIdentity(
            command_id=23,
            method_name="prepare",
            requires_ack=True,
            engine_step_id=7,
            repeat_index=3,
            request_set_sha256="a" * 64,
            batch_kind="decode",
            speculative_selected_sequence_ids_sha256="b" * 64,
            dispatch_started_monotonic_ns=300,
            dispatch_published_monotonic_ns=400,
        ),
    )
    runner = types.SimpleNamespace(
        world_size=2,
        rank=0,
        shm=shared,
        event=[event],
    )

    write_shm(runner, envelope)

    assert seen == [envelope]
    assert seen[0].trace_identity.dispatch_published_monotonic_ns == 400


def test_model_runner_read_records_wake_and_read_before_return():
    read_shm = _model_runner_method("read_shm")
    timeline = _Timeline(enabled=True)
    shared = _Buffer()
    envelope = ModelRunnerCommandEnvelope(
        command_id=24,
        method_name="prepare",
        args=(),
        requires_ack=True,
        trace_identity=CommandTraceIdentity(
            command_id=24,
            method_name="prepare",
            requires_ack=True,
            engine_step_id=7,
            repeat_index=3,
            request_set_sha256="a" * 64,
            batch_kind="decode",
            speculative_selected_sequence_ids_sha256="b" * 64,
            dispatch_started_monotonic_ns=300,
            dispatch_published_monotonic_ns=400,
        ),
    )
    payload = pickle.dumps(envelope)
    shared.buf[0:4] = len(payload).to_bytes(4, "little")
    shared.buf[4:4 + len(payload)] = payload
    worker = types.SimpleNamespace(
        world_size=2,
        rank=1,
        shm=shared,
        event=_Event(),
        _command_ids=count(),
        command_timeline=timeline,
        _command_timeline_clock_ns=iter((500, 600)).__next__,
    )

    assert read_shm(worker) == envelope
    assert timeline.receives == [(envelope.trace_identity, 500, 600)]


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
        command_timeline = _Timeline(enabled=False)
        _command_timeline_clock_ns = staticmethod(lambda: 0)

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
    assert "ModelRunnerCommandTimelineRecorder.disabled" in source
    assert "self._command_timeline_clock_ns = time.monotonic_ns" in source


def test_model_runner_command_timeline_lifecycle_resets_rows():
    configure = _model_runner_method("configure_command_timeline")
    reset = _model_runner_method("reset_command_timeline")
    snapshot = _model_runner_method("command_timeline_snapshot")
    runner = types.SimpleNamespace(
        rank=2,
        command_timeline=ModelRunnerCommandTimelineRecorder.disabled(2),
        _command_timeline_max_rows=8,
    )

    assert configure(runner, True, 16) == {
        "rank": 2,
        "enabled": True,
        "max_rows": 16,
    }
    assert snapshot(runner) == {
        "schema_version": 1,
        "rank": 2,
        "enabled": True,
        "clock": {
            "boot_id": "boot",
            "implementation": "clock_gettime(CLOCK_MONOTONIC)",
            "resolution_s": 1e-9,
            "monotonic": True,
            "adjustable": False,
            "captured_at_unix_ns": 1,
        },
        "rows": [],
        "dropped_rows": 0,
    }
    runner.configure_command_timeline = lambda enabled, max_rows: configure(
        runner,
        enabled,
        max_rows,
    )
    assert reset(runner) == {
        "rank": 2,
        "enabled": True,
        "max_rows": 16,
    }
    assert snapshot(runner)["rows"] == []


def test_engine_tp1_acknowledged_call_is_local_only():
    call_ack = _engine_method("call_model_runner_acknowledged")
    dispatched = []

    class Runner:
        world_size = 1
        command_timeline = _Timeline(enabled=True)

        def dispatch_command(self, method_name, *args, requires_ack):
            envelope = ModelRunnerCommandEnvelope(
                command_id=41,
                method_name=method_name,
                args=tuple(args),
                requires_ack=requires_ack,
            )
            dispatched.append(envelope)
            return envelope

        @staticmethod
        def execute_command_envelope(envelope):
            return sum(envelope.args)

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=None,
        ps=[],
        _clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("TP1 recorded a worker ack wait")
        ),
    )

    assert call_ack(
        engine,
        "add",
        4,
        6,
        timeout_s=1.0,
    ) == (10, ())
    assert dispatched[0].requires_ack is False
    assert engine.model_runner.command_timeline.ack_waits == []


def test_engine_tp1_traced_local_call_finishes_without_ack_wait():
    dispatch = _model_runner_method("dispatch_command")
    execute = _model_runner_method("execute_command_envelope")
    call_ack = _engine_method("call_model_runner_acknowledged")
    timeline = ModelRunnerCommandTimelineRecorder(
        rank=0,
        max_rows=8,
        clock_identity=CommandClockIdentity(
            boot_id="boot",
            implementation="clock_gettime(CLOCK_MONOTONIC)",
            resolution_s=1e-9,
            monotonic=True,
            adjustable=False,
            captured_at_unix_ns=1,
        ),
    )
    runner = types.SimpleNamespace(
        rank=0,
        world_size=1,
        _command_ids=count(),
        command_timeline=timeline,
        _command_timeline_clock_ns=iter((10, 20, 30, 40)).__next__,
        _active_command_timeline_trace=_step_trace,
        write_shm=lambda envelope: None,
        add=lambda left, right: left + right,
    )
    runner.dispatch_command = types.MethodType(dispatch, runner)
    runner.execute_command_envelope = types.MethodType(execute, runner)
    engine = types.SimpleNamespace(
        model_runner=runner,
        model_runner_ack_collector=None,
        ps=[],
        _clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("TP1 recorded a worker ack wait")
        ),
    )

    assert call_ack(
        engine,
        "add",
        4,
        6,
        timeout_s=1.0,
    ) == (10, ())
    row = timeline.snapshot()["rows"][0]
    assert row["requires_ack"] is False
    assert row["ack_wait_started_monotonic_ns"] is None
    assert row["ack_wait_finished_monotonic_ns"] is None


def test_engine_tp_acknowledged_call_collects_workers():
    call_ack = _engine_method("call_model_runner_acknowledged")
    rank_alive = _engine_method("_is_worker_rank_alive")
    envelope = ModelRunnerCommandEnvelope(
        command_id=8,
        method_name="prepare",
        args=(17,),
        requires_ack=True,
        trace_identity=CommandTraceIdentity(
            command_id=8,
            method_name="prepare",
            requires_ack=True,
            engine_step_id=7,
            repeat_index=3,
            request_set_sha256="a" * 64,
            batch_kind="decode",
            speculative_selected_sequence_ids_sha256="b" * 64,
            dispatch_started_monotonic_ns=10,
            dispatch_published_monotonic_ns=20,
        ),
    )
    collector_calls = []

    class Runner:
        world_size = 3
        command_timeline = _Timeline(enabled=True)

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

        def execute_command_envelope(self, dispatched_envelope):
            assert dispatched_envelope is envelope
            value, = dispatched_envelope.args
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
        _clock_ns=iter((100, 200)).__next__,
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
    assert engine.model_runner.command_timeline.ack_waits == [
        (8, 100, 200)
    ]


def test_engine_command_timeline_management_is_acknowledged_all_rank():
    configure = _engine_method("configure_command_timeline")
    reset = _engine_method("reset_command_timeline")
    snapshots = _engine_method("command_timeline_snapshots")
    calls = []

    def call_ack(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        if method_name == "configure_command_timeline":
            local = {"rank": 0, "enabled": args[0], "max_rows": args[1]}
            workers = tuple(
                types.SimpleNamespace(
                    rank=rank,
                    result={
                        "rank": rank,
                        "enabled": args[0],
                        "max_rows": args[1],
                    },
                )
                for rank in (1, 2)
            )
            return local, workers
        if method_name == "reset_command_timeline":
            local = {"rank": 0, "enabled": True, "max_rows": 64}
            workers = tuple(
                types.SimpleNamespace(
                    rank=rank,
                    result={"rank": rank, "enabled": True, "max_rows": 64},
                )
                for rank in (1, 2)
            )
            return local, workers
        local = {"rank": 0, "enabled": True, "rows": [{"command_id": 5}]}
        workers = tuple(
            types.SimpleNamespace(
                rank=rank,
                result={
                    "rank": rank,
                    "enabled": True,
                    "rows": [{"command_id": 5}],
                },
            )
            for rank in (1, 2)
        )
        return local, workers

    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=3),
        call_model_runner_acknowledged=call_ack,
    )

    assert configure(
        engine,
        True,
        64,
        timeout_s=4.0,
    ) == {
        "enabled": True,
        "max_rows": 64,
        "rank_inventory": [0, 1, 2],
    }
    assert reset(engine, timeout_s=5.0) == (
        {"rank": 0, "enabled": True, "max_rows": 64},
        {"rank": 1, "enabled": True, "max_rows": 64},
        {"rank": 2, "enabled": True, "max_rows": 64},
    )
    measured = snapshots(engine, timeout_s=6.0)
    assert tuple(row["rank"] for row in measured) == (0, 1, 2)
    assert all(
        snapshot["rows"] == [{"command_id": 5}]
        for snapshot in measured
    )
    assert calls == [
        ("configure_command_timeline", (True, 64), 4.0),
        ("reset_command_timeline", (), 5.0),
        ("command_timeline_snapshot", (), 6.0),
    ]


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

        def execute_command_envelope(self, envelope):
            return self.fail(*envelope.args)

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


def test_engine_traced_local_exception_terminalizes_timeline():
    dispatch = _model_runner_method("dispatch_command")
    execute = _model_runner_method("execute_command_envelope")
    call_ack = _engine_method("call_model_runner_acknowledged")
    timeline = ModelRunnerCommandTimelineRecorder(
        rank=0,
        max_rows=8,
        clock_identity=CommandClockIdentity(
            boot_id="boot",
            implementation="clock_gettime(CLOCK_MONOTONIC)",
            resolution_s=1e-9,
            monotonic=True,
            adjustable=False,
            captured_at_unix_ns=1,
        ),
    )
    runner = types.SimpleNamespace(
        rank=0,
        world_size=2,
        _command_ids=count(29),
        command_timeline=timeline,
        _command_timeline_clock_ns=iter((10, 20, 30, 40)).__next__,
        _active_command_timeline_trace=_step_trace,
        write_shm=lambda envelope: None,
        fail=lambda: (_ for _ in ()).throw(ValueError("rank zero failed")),
    )
    runner.dispatch_command = types.MethodType(dispatch, runner)
    runner.execute_command_envelope = types.MethodType(execute, runner)
    poison_reasons = []
    collector = types.SimpleNamespace(
        poison=poison_reasons.append,
    )
    engine = types.SimpleNamespace(
        model_runner=runner,
        model_runner_ack_collector=collector,
        ps=[_Process()],
        _clock_ns=iter((50,)).__next__,
    )

    try:
        call_ack(engine, "fail", timeout_s=1.0)
    except ValueError as error:
        assert str(error) == "rank zero failed"
    else:
        raise AssertionError("rank-zero exception was swallowed")

    row = timeline.snapshot()["rows"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "ValueError"
    assert row["error_detail"] == "rank zero failed"
    assert row["terminal_error_monotonic_ns"] == 50
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

        def execute_command_envelope(self, envelope):
            return self.prepare(*envelope.args)

    class Collector:
        def collect(self, *args, **kwargs):
            raise TimeoutError("worker ack timeout")

    engine = types.SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=Collector(),
        ps=[_Process()],
        _is_worker_rank_alive=lambda rank: True,
        _clock_ns=lambda: 100,
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


def test_engine_traced_collector_failure_terminalizes_ack_wait():
    dispatch = _model_runner_method("dispatch_command")
    execute = _model_runner_method("execute_command_envelope")
    call_ack = _engine_method("call_model_runner_acknowledged")

    class Collector:
        def __init__(self, error):
            self.error = error

        def collect(self, *args, **kwargs):
            raise self.error

    cases = (
        (39, TimeoutError("worker ack timeout")),
        (49, RuntimeError("worker error acknowledgement")),
    )
    for command_id, expected_error in cases:
        timeline = ModelRunnerCommandTimelineRecorder(
            rank=0,
            max_rows=8,
            clock_identity=CommandClockIdentity(
                boot_id="boot",
                implementation="clock_gettime(CLOCK_MONOTONIC)",
                resolution_s=1e-9,
                monotonic=True,
                adjustable=False,
                captured_at_unix_ns=1,
            ),
        )
        runner = types.SimpleNamespace(
            rank=0,
            world_size=2,
            _command_ids=count(command_id),
            command_timeline=timeline,
            _command_timeline_clock_ns=iter((10, 20, 30, 40)).__next__,
            _active_command_timeline_trace=_step_trace,
            write_shm=lambda envelope: None,
            prepare=lambda: "local-ok",
        )
        runner.dispatch_command = types.MethodType(dispatch, runner)
        runner.execute_command_envelope = types.MethodType(execute, runner)
        engine = types.SimpleNamespace(
            model_runner=runner,
            model_runner_ack_collector=Collector(expected_error),
            ps=[_Process()],
            _is_worker_rank_alive=lambda rank: True,
            _clock_ns=iter((50, 60)).__next__,
        )

        try:
            call_ack(engine, "prepare", timeout_s=1.0)
        except (TimeoutError, RuntimeError) as error:
            assert error is expected_error
        else:
            raise AssertionError("collector failure was swallowed")

        row = timeline.snapshot()["rows"][0]
        assert row["status"] == "error"
        assert row["error_type"] == type(expected_error).__name__
        assert row["error_detail"] == str(expected_error)
        assert row["ack_wait_started_monotonic_ns"] == 50
        assert row["ack_wait_finished_monotonic_ns"] == 60
        assert row["terminal_error_monotonic_ns"] == 60


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
