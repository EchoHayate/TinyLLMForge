import importlib.util
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    if package_name in sys.modules:
        continue
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package


ack_module = _load_module(
    "tinyvllm.engine.model_runner_command_ack",
    "tinyvllm/engine/model_runner_command_ack.py",
)

ModelRunnerCommandEnvelope = ack_module.ModelRunnerCommandEnvelope
ModelRunnerCommandAck = ack_module.ModelRunnerCommandAck
ModelRunnerCommandAckCollector = (
    ack_module.ModelRunnerCommandAckCollector
)
execute_acknowledged_command = (
    ack_module.execute_acknowledged_command
)


class _Target:
    def add(self, left, right):
        return left + right

    def fail(self, message):
        raise ValueError(message)

    def stop(self):
        raise SystemExit(9)


class _Recorder:
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)


class _SendFailure:
    def __call__(self, value):
        raise OSError("ack pipe closed")


class _ScriptedReceiver:
    def __init__(self, values=(), *, recv_error=None):
        self.values = list(values)
        self.recv_error = recv_error

    def poll(self, timeout=0.0):
        return bool(self.values) or self.recv_error is not None

    def recv(self):
        if self.recv_error is not None:
            raise self.recv_error
        return self.values.pop(0)


class _NeverReadyReceiver:
    def poll(self, timeout=0.0):
        return False

    def recv(self):
        raise AssertionError("recv called without readiness")


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def _worker_main(
    rank,
    envelope,
    send_connection,
    delay_s,
):
    if delay_s:
        time.sleep(delay_s)
    try:
        execute_acknowledged_command(
            envelope,
            rank=rank,
            target=_Target(),
            send_ack=send_connection.send,
        )
    finally:
        send_connection.close()


def _exit_without_ack():
    return None


def _spawn_worker(context, rank, envelope, delay_s=0.0):
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_main,
        args=(
            rank,
            envelope,
            send_connection,
            delay_s,
        ),
    )
    process.start()
    send_connection.close()
    return process, receive_connection


def test_envelope_and_ack_validate_and_pickle_round_trip():
    envelope = ModelRunnerCommandEnvelope(
        command_id=7,
        method_name="add",
        args=(2, 3),
        requires_ack=True,
    )
    acknowledgement = ModelRunnerCommandAck(
        command_id=7,
        rank=1,
        status="ok",
        result=5,
    )

    assert pickle.loads(pickle.dumps(envelope)) == envelope
    assert pickle.loads(pickle.dumps(acknowledgement)) == acknowledgement

    invalid_values = (
        lambda: ModelRunnerCommandEnvelope(
            command_id=-1,
            method_name="add",
            args=(),
            requires_ack=True,
        ),
        lambda: ModelRunnerCommandEnvelope(
            command_id=1,
            method_name="",
            args=(),
            requires_ack=True,
        ),
        lambda: ModelRunnerCommandEnvelope(
            command_id=1,
            method_name="_private",
            args=(),
            requires_ack=True,
        ),
        lambda: ModelRunnerCommandEnvelope(
            command_id=1,
            method_name="add",
            args=[],
            requires_ack=True,
        ),
        lambda: ModelRunnerCommandAck(
            command_id=1,
            rank=1,
            status="unknown",
        ),
    )
    for build in invalid_values:
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid command value was accepted")


def test_executor_acknowledged_success_and_bounded_error():
    recorder = _Recorder()
    result = execute_acknowledged_command(
        ModelRunnerCommandEnvelope(
            command_id=3,
            method_name="add",
            args=(4, 5),
            requires_ack=True,
        ),
        rank=2,
        target=_Target(),
        send_ack=recorder,
    )

    assert result == 9
    assert recorder.values == [
        ModelRunnerCommandAck(
            command_id=3,
            rank=2,
            status="ok",
            result=9,
        )
    ]

    long_message = "x" * 5000
    result = execute_acknowledged_command(
        ModelRunnerCommandEnvelope(
            command_id=4,
            method_name="fail",
            args=(long_message,),
            requires_ack=True,
        ),
        rank=2,
        target=_Target(),
        send_ack=recorder,
    )

    assert result is None
    error_ack = recorder.values[-1]
    assert error_ack.status == "error"
    assert error_ack.error_type == "ValueError"
    assert len(error_ack.error_detail.encode("utf-8")) <= 4096
    assert error_ack.result is None


def test_executor_fire_and_forget_and_send_failure_semantics():
    recorder = _Recorder()
    envelope = ModelRunnerCommandEnvelope(
        command_id=5,
        method_name="add",
        args=(6, 7),
        requires_ack=False,
    )

    assert execute_acknowledged_command(
        envelope,
        rank=1,
        target=_Target(),
        send_ack=recorder,
    ) == 13
    assert recorder.values == []

    try:
        execute_acknowledged_command(
            ModelRunnerCommandEnvelope(
                command_id=6,
                method_name="fail",
                args=("fire-and-forget",),
                requires_ack=False,
            ),
            rank=1,
            target=_Target(),
            send_ack=recorder,
        )
    except ValueError as error:
        assert str(error) == "fire-and-forget"
    else:
        raise AssertionError("fire-and-forget exception was swallowed")

    try:
        execute_acknowledged_command(
            ModelRunnerCommandEnvelope(
                command_id=7,
                method_name="add",
                args=(1, 2),
                requires_ack=True,
            ),
            rank=1,
            target=_Target(),
            send_ack=_SendFailure(),
        )
    except OSError as error:
        assert "ack pipe closed" in str(error)
    else:
        raise AssertionError("ack send failure was swallowed")

    try:
        execute_acknowledged_command(
            ModelRunnerCommandEnvelope(
                command_id=8,
                method_name="stop",
                args=(),
                requires_ack=True,
            ),
            rank=1,
            target=_Target(),
            send_ack=recorder,
        )
    except SystemExit as error:
        assert error.code == 9
    else:
        raise AssertionError("BaseException was converted to acknowledgement")


def test_collector_orders_real_spawned_worker_success():
    context = mp.get_context("spawn")
    envelope = ModelRunnerCommandEnvelope(
        command_id=10,
        method_name="add",
        args=(20, 22),
        requires_ack=True,
    )
    worker_one, receiver_one = _spawn_worker(
        context,
        1,
        envelope,
        delay_s=0.1,
    )
    worker_two, receiver_two = _spawn_worker(
        context,
        2,
        envelope,
        delay_s=0.0,
    )
    collector = ModelRunnerCommandAckCollector((
        (1, receiver_one),
        (2, receiver_two),
    ))
    workers = {
        1: worker_one,
        2: worker_two,
    }

    acknowledgements = collector.collect(
        envelope.command_id,
        expected_ranks=(1, 2),
        timeout_s=3.0,
        is_rank_alive=lambda rank: workers[rank].is_alive(),
    )

    assert tuple(ack.rank for ack in acknowledgements) == (1, 2)
    assert tuple(ack.result for ack in acknowledgements) == (42, 42)
    for process in workers.values():
        process.join(timeout=2.0)
        assert process.exitcode == 0
    receiver_one.close()
    receiver_two.close()


def test_collector_worker_error_poison_and_reuse_rejected():
    receiver = _ScriptedReceiver((
        ModelRunnerCommandAck(
            command_id=11,
            rank=1,
            status="error",
            error_type="ValueError",
            error_detail="rank failed",
        ),
    ))
    collector = ModelRunnerCommandAckCollector(((1, receiver),))

    try:
        collector.collect(
            11,
            expected_ranks=(1,),
            timeout_s=1.0,
            is_rank_alive=lambda rank: True,
        )
    except RuntimeError as error:
        assert "rank failed" in str(error)
    else:
        raise AssertionError("worker error acknowledgement was accepted")
    assert collector.poisoned
    try:
        collector.collect(
            12,
            expected_ranks=(1,),
            timeout_s=1.0,
            is_rank_alive=lambda rank: True,
        )
    except RuntimeError as error:
        assert "poisoned" in str(error)
    else:
        raise AssertionError("poisoned collector was reused")


def test_collector_public_poison_is_idempotent_and_validated():
    collector = ModelRunnerCommandAckCollector((
        (1, _NeverReadyReceiver()),
    ))

    collector.poison("rank zero failed")
    collector.poison("later failure")

    assert collector.poisoned
    try:
        collector.collect(
            101,
            expected_ranks=(1,),
            timeout_s=1.0,
            is_rank_alive=lambda rank: True,
        )
    except RuntimeError as error:
        assert "rank zero failed" in str(error)
        assert "later failure" not in str(error)
    else:
        raise AssertionError("externally poisoned collector was reused")

    fresh = ModelRunnerCommandAckCollector((
        (1, _NeverReadyReceiver()),
    ))
    for invalid in ("", None):
        try:
            fresh.poison(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid poison reason was accepted")
    assert fresh.poisoned is False


def test_collector_real_spawned_worker_error_is_explicit():
    context = mp.get_context("spawn")
    envelope = ModelRunnerCommandEnvelope(
        command_id=111,
        method_name="fail",
        args=("real worker failure",),
        requires_ack=True,
    )
    worker, receiver = _spawn_worker(
        context,
        1,
        envelope,
    )
    collector = ModelRunnerCommandAckCollector(((1, receiver),))

    try:
        collector.collect(
            envelope.command_id,
            expected_ranks=(1,),
            timeout_s=3.0,
            is_rank_alive=lambda rank: worker.is_alive(),
        )
    except RuntimeError as error:
        assert "real worker failure" in str(error)
    else:
        raise AssertionError("real worker error was accepted")
    assert collector.poisoned
    worker.join(timeout=2.0)
    assert worker.exitcode == 0
    receiver.close()


def test_collector_outer_ok_preserves_inner_restore_miss():
    inner_restore_ack = {
        "ticket_id": 41,
        "participant_id": 1,
        "status": "miss",
        "detail": "",
    }
    receiver = _ScriptedReceiver((
        ModelRunnerCommandAck(
            command_id=112,
            rank=1,
            status="ok",
            result=inner_restore_ack,
        ),
    ))
    collector = ModelRunnerCommandAckCollector(((1, receiver),))

    acknowledgements = collector.collect(
        112,
        expected_ranks=(1,),
        timeout_s=1.0,
        is_rank_alive=lambda rank: True,
    )

    assert acknowledgements[0].status == "ok"
    assert acknowledgements[0].result["status"] == "miss"
    assert collector.poisoned is False


def test_collector_timeout_uses_one_deadline_and_reports_missing_ranks():
    clock = _FakeClock()
    collector = ModelRunnerCommandAckCollector(
        (
            (1, _NeverReadyReceiver()),
            (2, _NeverReadyReceiver()),
        ),
        clock=clock,
        sleeper=clock.sleep,
        poll_interval_s=0.05,
    )

    try:
        collector.collect(
            12,
            expected_ranks=(1, 2),
            timeout_s=0.2,
            is_rank_alive=lambda rank: True,
        )
    except TimeoutError as error:
        assert "1, 2" in str(error)
    else:
        raise AssertionError("missing workers did not time out")
    assert 0.2 <= clock.value < 0.3
    assert collector.poisoned


def test_collector_real_spawned_worker_timeout_is_fail_closed():
    context = mp.get_context("spawn")
    envelope = ModelRunnerCommandEnvelope(
        command_id=114,
        method_name="add",
        args=(1, 2),
        requires_ack=True,
    )
    worker, receiver = _spawn_worker(
        context,
        1,
        envelope,
        delay_s=1.0,
    )
    collector = ModelRunnerCommandAckCollector(((1, receiver),))

    started = time.monotonic()
    try:
        collector.collect(
            envelope.command_id,
            expected_ranks=(1,),
            timeout_s=0.1,
            is_rank_alive=lambda rank: worker.is_alive(),
        )
    except TimeoutError as error:
        assert "missing ranks: 1" in str(error)
    else:
        raise AssertionError("real delayed worker did not time out")
    elapsed = time.monotonic() - started
    assert 0.08 <= elapsed < 0.8
    assert collector.poisoned
    worker.terminate()
    worker.join(timeout=2.0)
    receiver.close()


def test_collector_detects_dead_worker_before_timeout():
    clock = _FakeClock()
    collector = ModelRunnerCommandAckCollector(
        ((1, _NeverReadyReceiver()),),
        clock=clock,
        sleeper=clock.sleep,
        poll_interval_s=0.05,
    )

    try:
        collector.collect(
            13,
            expected_ranks=(1,),
            timeout_s=5.0,
            is_rank_alive=lambda rank: False,
        )
    except RuntimeError as error:
        assert "not alive" in str(error)
    else:
        raise AssertionError("dead worker was not detected")
    assert clock.value == 0.0
    assert collector.poisoned


def test_collector_detects_real_spawned_worker_death():
    context = mp.get_context("spawn")
    process = context.Process(target=_exit_without_ack)
    process.start()
    process.join(timeout=2.0)
    assert process.exitcode == 0
    collector = ModelRunnerCommandAckCollector((
        (1, _NeverReadyReceiver()),
    ))

    try:
        collector.collect(
            113,
            expected_ranks=(1,),
            timeout_s=3.0,
            is_rank_alive=lambda rank: process.is_alive(),
        )
    except RuntimeError as error:
        assert "not alive" in str(error)
    else:
        raise AssertionError("real worker death was not detected")
    assert collector.poisoned


def test_collector_rejects_stale_wrong_rank_malformed_and_receive_failure():
    cases = (
        (
            _ScriptedReceiver((
                ModelRunnerCommandAck(
                    command_id=15,
                    rank=1,
                    status="ok",
                ),
            )),
            "command",
        ),
        (
            _ScriptedReceiver((
                ModelRunnerCommandAck(
                    command_id=14,
                    rank=2,
                    status="ok",
                ),
            )),
            "rank",
        ),
        (
            _ScriptedReceiver(("not-an-ack",)),
            "acknowledgement",
        ),
        (
            _ScriptedReceiver(
                recv_error=EOFError("closed pipe"),
            ),
            "receive",
        ),
    )
    for receiver, fragment in cases:
        collector = ModelRunnerCommandAckCollector(((1, receiver),))
        try:
            collector.collect(
                14,
                expected_ranks=(1,),
                timeout_s=1.0,
                is_rank_alive=lambda rank: True,
            )
        except RuntimeError as error:
            assert fragment in str(error).lower()
        else:
            raise AssertionError(
                f"collector accepted invalid case: {fragment}"
            )
        assert collector.poisoned


def test_collector_validates_receiver_and_collection_contract():
    invalid_builds = (
        lambda: ModelRunnerCommandAckCollector(()),
        lambda: ModelRunnerCommandAckCollector((
            (1, _NeverReadyReceiver()),
            (1, _NeverReadyReceiver()),
        )),
        lambda: ModelRunnerCommandAckCollector((
            (0, _NeverReadyReceiver()),
        )),
        lambda: ModelRunnerCommandAckCollector((
            (1, _NeverReadyReceiver()),
            (3, _NeverReadyReceiver()),
        )),
        lambda: ModelRunnerCommandAckCollector((
            (2, _NeverReadyReceiver()),
        )),
    )
    for build in invalid_builds:
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid collector receivers were accepted")

    collector = ModelRunnerCommandAckCollector((
        (1, _NeverReadyReceiver()),
    ))
    invalid_calls = (
        {"command_id": -1, "expected_ranks": (1,), "timeout_s": 1.0},
        {"command_id": 1, "expected_ranks": (), "timeout_s": 1.0},
        {"command_id": 1, "expected_ranks": (2,), "timeout_s": 1.0},
        {"command_id": 1, "expected_ranks": (1,), "timeout_s": 0.0},
    )
    for values in invalid_calls:
        try:
            collector.collect(
                is_rank_alive=lambda rank: True,
                **values,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid collection request was accepted")
    assert collector.poisoned is False


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "model runner command acknowledgement tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
