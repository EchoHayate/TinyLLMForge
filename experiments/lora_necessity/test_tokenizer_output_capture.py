import importlib
import importlib.util
import threading
import unittest
from unittest.mock import patch


STREAMS = ("result", "stdout", "stderr")
LIMITS = (67_108_864, 1_048_576, 1_048_576)
FALSE_FLAGS = (
    "resource_limits_enforced",
    "root_write_limit_enforced",
    "deadline_enforced",
    "process_termination_enforced",
    "cleanup_verified",
    "benchmark_eligible",
    "formal_evaluation_ready",
    "net_benefit_established",
    "isolated",
    "all_possible_second_rounds_validated",
)


class HostileValue:
    def __eq__(self, other):
        raise AssertionError("unexpected caller operation")

    def __hash__(self):
        raise AssertionError("unexpected caller operation")

    def __str__(self):
        raise AssertionError("unexpected caller operation")

    def __repr__(self):
        raise AssertionError("unexpected caller operation")

    def __len__(self):
        raise AssertionError("unexpected caller operation")

    def __bytes__(self):
        raise AssertionError("unexpected caller operation")


class StringSubclass(HostileValue, str):
    pass


class BytesSubclass(HostileValue, bytes):
    pass


class StorageFault(bytearray):
    def __init__(self, initial, *, append_error=None, copy_error=None, prefix=0):
        super().__init__(initial)
        self.append_error = append_error
        self.copy_error = copy_error
        self.prefix = prefix
        self.append_calls = 0
        self.copy_calls = 0

    def extend(self, chunk):
        self.append_calls += 1
        if self.append_error is not None:
            super().extend(chunk[:self.prefix])
            raise self.append_error
        super().extend(chunk)

    def __bytes__(self):
        self.copy_calls += 1
        if self.copy_error is not None:
            raise self.copy_error
        return bytes(memoryview(self))


class CaptureCancellation(BaseException):
    pass


class OutputCaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("tokenizer_output_capture"),
            "capture implementation missing",
        )
        self.module = importlib.import_module("tokenizer_output_capture")

    def complete(self, capture):
        for stream in STREAMS:
            self.assertIsNone(capture.end(stream))
        self.assertIsNone(capture.finish())

    def assert_invalid(self, operation, *arguments):
        self.assertTrue(
            hasattr(self.module, "OutputCaptureError"), "capture error missing"
        )
        with self.assertRaises(BaseException) as caught:
            operation(*arguments)
        error = caught.exception
        self.assertTrue(
            type(error) is self.module.OutputCaptureError,
            "capture error type mismatch",
        )
        self.assertTrue(isinstance(error, ValueError), "capture error base mismatch")
        self.assertTrue(
            str(error) == "tokenizer output capture invalid",
            "capture error message mismatch",
        )
        self.assertTrue(
            error.args == ("tokenizer output capture invalid",),
            "capture error arguments mismatch",
        )
        return error

    def assert_failure(self, capture, reason, stream):
        snapshot = capture.snapshot()
        self.assertEqual(snapshot["state"], "FAILED")
        self.assertFalse(snapshot["capture_complete"])
        self.assertTrue(
            snapshot["failure"] == {"reason": reason, "stream": stream},
            "failure metadata mismatch",
        )
        for flag in FALSE_FLAGS:
            self.assertIs(snapshot[flag], False)

    def fill(self, capture, stream, count):
        chunk = bytes(range(256)) * 256
        whole, remainder = divmod(count, len(chunk))
        for _ in range(whole):
            capture.feed(stream, chunk)
        if remainder:
            capture.feed(stream, chunk[:remainder])

    def test_construction_snapshot_schema(self):
        capture = self.module.OutputCapture()
        expected = {
            "kind": "LOCAL_TOKENIZER_OUTPUT_CAPTURE_ONLY",
            "state": "OPEN",
            "streams": [
                {
                    "stream": stream,
                    "limit_bytes": limit,
                    "accepted_bytes": 0,
                    "eof": False,
                }
                for stream, limit in zip(STREAMS, LIMITS)
            ],
            "total_accepted_bytes": 0,
            "capture_complete": False,
            "failure": None,
            "cleanup_status": "CLEANUP_UNVERIFIED",
            **dict.fromkeys(FALSE_FLAGS, False),
        }
        self.assertTrue(capture.snapshot() == expected, "snapshot schema mismatch")

    def test_binary_preservation(self):
        capture = self.module.OutputCapture()
        payload = bytes((0, 255, 123, 125))
        self.assertIsNone(capture.feed("result", payload))
        self.complete(capture)
        outputs = capture.private_outputs()
        self.assertTrue(outputs["result"] == payload, "binary preservation mismatch")
        self.assertTrue(capture.snapshot()["capture_complete"])
        self.assertFalse(capture.snapshot()["benchmark_eligible"])

    def test_empty_completion_requires_explicit_finish(self):
        capture = self.module.OutputCapture()
        for stream in reversed(STREAMS):
            self.assertIsNone(capture.end(stream))
        self.assertEqual(capture.snapshot()["state"], "OPEN")
        self.assertFalse(capture.snapshot()["capture_complete"])
        self.assertIsNone(capture.finish())
        self.assertIsNone(capture.finish())
        self.assertEqual(capture.snapshot()["state"], "FINISHED")
        self.assertTrue(
            all(value == bytes() for value in capture.private_outputs().values()),
            "empty completion mismatch",
        )

    def test_interleaving_and_fragmentation_preserve_order(self):
        capture = self.module.OutputCapture()
        buffers = tuple(capture._buffers.values())
        fragments = (bytes((0,)), bytes((255,)), bytes((128,)))
        for _ in range(2_048):
            for stream, fragment in zip(STREAMS, fragments):
                capture.feed(stream, fragment)
        self.assertTrue(
            all(
                type(buffer) is bytearray and buffer is capture._buffers[stream]
                for stream, buffer in zip(STREAMS, buffers)
            ),
            "fragment storage structure mismatch",
        )
        self.complete(capture)
        outputs = capture.private_outputs()
        self.assertTrue(tuple(outputs) == STREAMS, "output order mismatch")
        self.assertEqual(capture.snapshot()["total_accepted_bytes"], 6_144)
        for stream, fragment in zip(STREAMS, fragments):
            self.assertTrue(
                outputs[stream] == fragment * 2_048, "fragment preservation mismatch"
            )

    def test_all_three_production_capacities(self):
        capture = self.module.OutputCapture()
        for stream, limit in zip(STREAMS, LIMITS):
            self.fill(capture, stream, limit)
        snapshot = capture.snapshot()
        self.assertEqual(snapshot["total_accepted_bytes"], 69_206_016)
        self.assertTrue(
            [record["accepted_bytes"] for record in snapshot["streams"]]
            == list(LIMITS),
            "capacity metadata mismatch",
        )
        self.assertTrue(
            all(record["eof"] is False for record in snapshot["streams"]),
            "capacity must not declare eof",
        )
        self.complete(capture)
        outputs = capture.private_outputs()
        expected_chunk = bytes(range(256)) * 256
        for stream, limit in zip(STREAMS, LIMITS):
            self.assertEqual(len(outputs[stream]), limit)
            self.assertTrue(type(outputs[stream]) is bytes, "output type mismatch")
            for offset in range(0, limit, 65_536):
                self.assertTrue(
                    outputs[stream][offset:offset + 65_536] == expected_chunk,
                    "capacity preservation mismatch",
                )

    def test_each_production_limit_one_byte_overflow(self):
        for stream, limit in zip(STREAMS, LIMITS):
            with self.subTest(stream=stream):
                capture = self.module.OutputCapture()
                self.fill(capture, stream, limit)
                self.assert_invalid(capture.feed, stream, bytes((1,)))
                self.assert_failure(capture, "OUTPUT_LIMIT", stream)
                self.assertEqual(capture.snapshot()["total_accepted_bytes"], limit)
                outputs = capture.private_outputs()
                self.assertEqual(len(outputs[stream]), limit)
                self.assertTrue(
                    outputs[stream][-65_536:] == bytes(range(256)) * 256,
                    "overflow preservation mismatch",
                )
                del outputs, capture

    def test_each_production_limit_rejects_whole_chunk(self):
        for stream, limit in zip(STREAMS, LIMITS):
            with self.subTest(stream=stream):
                capture = self.module.OutputCapture()
                self.fill(capture, stream, limit - 1)
                before = capture.snapshot()
                self.assert_invalid(capture.feed, stream, bytes((1, 2)))
                self.assert_failure(capture, "OUTPUT_LIMIT", stream)
                self.assertTrue(
                    capture.snapshot()["streams"] == before["streams"],
                    "whole chunk rejection mismatch",
                )
                self.assertEqual(
                    len(capture.private_outputs()[stream]), limit - 1
                )
                del capture

    def test_chunk_size_precedes_capacity(self):
        capture = self.module.OutputCapture()
        self.fill(capture, "stdout", LIMITS[1])
        self.assert_invalid(capture.feed, "stdout", bytes(65_537))
        self.assert_failure(capture, "PROTOCOL_ERROR", "stdout")
        self.assertEqual(capture.snapshot()["total_accepted_bytes"], LIMITS[1])

    def test_invalid_exact_chunk_types_and_sizes(self):
        invalid = (
            bytes(), bytes(65_537), BytesSubclass(bytes((1,))),
            bytearray((1,)), memoryview(bytes((1,))), "invalid", None,
            False, 1, [], {}, HostileValue(),
        )
        for case, chunk in enumerate(invalid):
            for stream in STREAMS:
                with self.subTest(case=case, stream=stream):
                    capture = self.module.OutputCapture()
                    self.assert_invalid(capture.feed, stream, chunk)
                    self.assert_failure(capture, "PROTOCOL_ERROR", stream)
                    self.assertEqual(capture.snapshot()["total_accepted_bytes"], 0)

    def test_invalid_streams_never_invoke_caller_operations(self):
        invalid = (
            "", "unknown", "RESULT", "stdout ", StringSubclass("result"),
            bytes((1,)), None, False, 1, [], {}, HostileValue(),
        )
        for case, stream in enumerate(invalid):
            for method in ("feed", "end"):
                with self.subTest(case=case, method=method):
                    capture = self.module.OutputCapture()
                    arguments = (stream, HostileValue()) if method == "feed" else (stream,)
                    self.assert_invalid(getattr(capture, method), *arguments)
                    self.assert_failure(capture, "PROTOCOL_ERROR", None)
                    self.assertEqual(capture.snapshot()["total_accepted_bytes"], 0)

    def test_duplicate_eof_and_feed_after_eof(self):
        for stream in STREAMS:
            for method in ("feed", "end"):
                with self.subTest(stream=stream, method=method):
                    capture = self.module.OutputCapture()
                    capture.feed(stream, bytes((1,)))
                    capture.end(stream)
                    before = capture.snapshot()["streams"]
                    arguments = (stream, HostileValue()) if method == "feed" else (stream,)
                    self.assert_invalid(getattr(capture, method), *arguments)
                    self.assert_failure(capture, "PROTOCOL_ERROR", stream)
                    self.assertTrue(
                        capture.snapshot()["streams"] == before,
                        "eof rejection mutation",
                    )

    def test_finish_requires_every_eof(self):
        for eof_mask in range(7):
            with self.subTest(case=eof_mask):
                capture = self.module.OutputCapture()
                for position, stream in enumerate(STREAMS):
                    if eof_mask & (1 << position):
                        capture.end(stream)
                before = capture.snapshot()["streams"]
                self.assert_invalid(capture.finish)
                self.assert_failure(capture, "PROTOCOL_ERROR", None)
                self.assertTrue(
                    capture.snapshot()["streams"] == before, "finish rejection mutation"
                )

    def test_private_outputs_open_latches_failure(self):
        capture = self.module.OutputCapture()
        capture.feed("stderr", bytes((1,)))
        self.assert_invalid(capture.private_outputs)
        self.assert_failure(capture, "PROTOCOL_ERROR", None)
        self.assertTrue(
            capture.private_outputs()["stderr"] == bytes((1,)),
            "private recovery mismatch",
        )

    def test_feed_and_end_after_finish_invalidate_capture(self):
        for method in ("feed", "end"):
            for case, stream in enumerate((*STREAMS, HostileValue())):
                with self.subTest(method=method, case=case):
                    capture = self.module.OutputCapture()
                    self.complete(capture)
                    arguments = (stream, HostileValue()) if method == "feed" else (stream,)
                    self.assert_invalid(getattr(capture, method), *arguments)
                    known = stream if type(stream) is str else None
                    self.assert_failure(capture, "PROTOCOL_ERROR", known)
                    self.assertEqual(capture.snapshot()["total_accepted_bytes"], 0)

    def test_abort_preserves_bytes_and_eofs_from_open_and_finished(self):
        self.assertTrue(
            callable(getattr(self.module.OutputCapture, "abort", None)),
            "abort implementation missing",
        )
        for finished in (False, True):
            with self.subTest(finished=finished):
                capture = self.module.OutputCapture()
                capture.feed("result", bytes((0, 255)))
                if finished:
                    self.complete(capture)
                else:
                    capture.end("stderr")
                before = capture.snapshot()["streams"]
                self.assertIsNone(capture.abort())
                self.assert_failure(capture, "ABORTED", None)
                self.assertTrue(
                    capture.snapshot()["streams"] == before, "abort mutation"
                )
                self.assertTrue(
                    capture.private_outputs()["result"] == bytes((0, 255)),
                    "abort preservation mismatch",
                )
                self.assertIsNone(capture.abort())

    def test_failed_state_keeps_first_failure_and_ignores_arguments(self):
        capture = self.module.OutputCapture()
        capture.feed("stderr", bytes((0, 255)))
        capture.end("stdout")
        self.assert_invalid(capture.feed, "result", bytes())
        before = capture.snapshot()
        for operation, arguments in (
            (capture.feed, (HostileValue(), HostileValue())),
            (capture.end, (HostileValue(),)),
            (capture.finish, ()),
        ):
            self.assert_invalid(operation, *arguments)
            self.assertTrue(capture.snapshot() == before, "first failure mutation")
        self.assertIsNone(capture.abort())
        self.assertTrue(capture.snapshot() == before, "failed abort mutation")
        self.assertTrue(
            capture.private_outputs()["stderr"] == bytes((0, 255)),
            "failed preservation mismatch",
        )

    def test_snapshot_and_private_outputs_are_independent(self):
        capture = self.module.OutputCapture()
        capture.feed("result", bytes((0, 255)))
        pristine = capture.snapshot()
        snapshot = capture.snapshot()
        snapshot["streams"][0]["accepted_bytes"] = -1
        snapshot["streams"].clear()
        snapshot["state"] = "changed"
        self.assertTrue(capture.snapshot() == pristine, "snapshot alias")
        self.complete(capture)
        first = capture.private_outputs()
        second = capture.private_outputs()
        self.assertIsNot(first, second)
        self.assertTrue(tuple(first) == STREAMS, "private output order mismatch")
        first["result"] = bytes((1,))
        first.clear()
        self.assertTrue(
            second["result"] == bytes((0, 255)), "private output alias"
        )
        capture.abort()
        snapshot = capture.snapshot()
        snapshot["failure"]["reason"] = "changed"
        snapshot["failure"]["stream"] = "changed"
        self.assert_failure(capture, "ABORTED", None)
        third = capture.private_outputs()
        self.assertIsNot(second, third)
        self.assertTrue(third == second, "terminal copy mismatch")

    def test_instances_own_separate_state_and_buffers(self):
        first = self.module.OutputCapture()
        second = self.module.OutputCapture()
        first.feed("result", bytes((1,)))
        first.end("stdout")
        self.assertEqual(second.snapshot()["total_accepted_bytes"], 0)
        self.assertTrue(
            all(not record["eof"] for record in second.snapshot()["streams"]),
            "instance eof alias",
        )
        self.complete(second)
        first.abort()
        self.assertEqual(second.snapshot()["state"], "FINISHED")

    def test_append_exceptions_latch_actual_retained_prefix(self):
        for stream in STREAMS:
            for prefix in (0, 1):
                for exception_type in (RuntimeError, MemoryError):
                    with self.subTest(
                        stream=stream, prefix=prefix, exception=exception_type.__name__
                    ):
                        capture = self.module.OutputCapture()
                        capture.feed(stream, bytes((0,)))
                        storage = StorageFault(
                            capture._buffers[stream],
                            append_error=exception_type("synthetic storage failure"),
                            prefix=prefix,
                        )
                        capture._buffers[stream] = storage
                        error = self.assert_invalid(
                            capture.feed, stream, bytes((255, 128))
                        )
                        self.assertTrue(error.__suppress_context__)
                        self.assertIsNone(error.__cause__)
                        self.assert_failure(capture, "CAPTURE_ERROR", stream)
                        self.assertEqual(storage.append_calls, 1)
                        self.assertEqual(
                            capture.snapshot()["total_accepted_bytes"], 1 + prefix
                        )
                        self.assertTrue(
                            capture.private_outputs()[stream]
                            == bytes((0, 255))[:1 + prefix],
                            "append prefix mismatch",
                        )
                        before = capture.snapshot()
                        self.assert_invalid(capture.feed, stream, bytes((1,)))
                        self.assertIsNone(capture.abort())
                        self.assertTrue(
                            capture.snapshot() == before, "append first failure mutation"
                        )
                        self.assertEqual(storage.append_calls, 1)

    def test_append_cancellation_preserves_original_object(self):
        for exception_type in (CaptureCancellation, KeyboardInterrupt, SystemExit):
            for prefix in (0, 1):
                with self.subTest(exception=exception_type.__name__, prefix=prefix):
                    capture = self.module.OutputCapture()
                    cancellation = exception_type("synthetic cancellation")
                    storage = StorageFault(
                        bytes((0,)), append_error=cancellation, prefix=prefix
                    )
                    capture._buffers["result"] = storage
                    with self.assertRaises(BaseException) as caught:
                        capture.feed("result", bytes((255, 128)))
                    self.assertTrue(
                        caught.exception is cancellation, "cancellation identity mismatch"
                    )
                    self.assert_failure(capture, "CAPTURE_ERROR", "result")
                    self.assertEqual(
                        capture.snapshot()["total_accepted_bytes"], 1 + prefix
                    )

    def test_copy_exceptions_are_atomic_and_preserve_first_failure(self):
        for stream in STREAMS:
            for failed in (False, True):
                for exception_type in (RuntimeError, MemoryError):
                    with self.subTest(
                        stream=stream, failed=failed, exception=exception_type.__name__
                    ):
                        capture = self.module.OutputCapture()
                        for name in STREAMS:
                            capture.feed(name, bytes((0, 255)))
                        self.complete(capture)
                        if failed:
                            self.assert_invalid(capture.end, "stdout")
                        before = capture.snapshot()
                        storage = StorageFault(
                            capture._buffers[stream],
                            copy_error=exception_type("synthetic storage failure"),
                        )
                        capture._buffers[stream] = storage
                        error = self.assert_invalid(capture.private_outputs)
                        self.assertTrue(error.__suppress_context__)
                        self.assertIsNone(error.__cause__)
                        self.assertEqual(storage.copy_calls, 1)
                        if failed:
                            self.assertTrue(
                                capture.snapshot() == before, "copy first failure mutation"
                            )
                        else:
                            self.assert_failure(capture, "CAPTURE_ERROR", None)
                        self.assertTrue(
                            capture.snapshot()["streams"] == before["streams"],
                            "copy retained metadata mismatch",
                        )
                        storage.copy_error = None
                        outputs = capture.private_outputs()
                        self.assertTrue(tuple(outputs) == STREAMS, "copy order mismatch")
                        self.assertTrue(
                            all(value == bytes((0, 255)) for value in outputs.values()),
                            "copy recovery mismatch",
                        )

    def test_copy_cancellation_preserves_original_object_and_first_failure(self):
        for exception_type in (CaptureCancellation, KeyboardInterrupt, SystemExit):
            for failed in (False, True):
                with self.subTest(exception=exception_type.__name__, failed=failed):
                    capture = self.module.OutputCapture()
                    capture.feed("stderr", bytes((0, 255)))
                    self.complete(capture)
                    if failed:
                        capture.abort()
                    before = capture.snapshot()
                    cancellation = exception_type("synthetic cancellation")
                    storage = StorageFault(
                        capture._buffers["stderr"], copy_error=cancellation
                    )
                    capture._buffers["stderr"] = storage
                    with self.assertRaises(BaseException) as caught:
                        capture.private_outputs()
                    self.assertTrue(
                        caught.exception is cancellation, "copy cancellation identity mismatch"
                    )
                    if failed:
                        self.assertTrue(
                            capture.snapshot() == before, "cancellation first failure mutation"
                        )
                    else:
                        self.assert_failure(capture, "CAPTURE_ERROR", None)
                    storage.copy_error = None
                    self.assertTrue(
                        capture.private_outputs()["stderr"] == bytes((0, 255)),
                        "cancellation preservation mismatch",
                    )

    def test_snapshot_and_repeated_finish_never_copy_or_sample_clock(self):
        capture = self.module.OutputCapture()
        capture.feed("result", bytes((0, 255)))
        storage = StorageFault(
            capture._buffers["result"], copy_error=RuntimeError("synthetic storage failure")
        )
        capture._buffers["result"] = storage
        with (
            patch("time.time", side_effect=AssertionError("unexpected clock operation")),
            patch("time.monotonic", side_effect=AssertionError("unexpected clock operation")),
            patch("time.monotonic_ns", side_effect=AssertionError("unexpected clock operation")),
            patch("time.perf_counter", side_effect=AssertionError("unexpected clock operation")),
        ):
            self.module.OutputCapture().snapshot()
            capture.snapshot()
            self.complete(capture)
            self.assertIsNone(capture.finish())
            capture.snapshot()
            capture.abort()
            capture.snapshot()
        self.assertEqual(storage.copy_calls, 0)
        self.assertEqual(storage.append_calls, 0)

    def run_in_worker(self, operation, *arguments):
        result = []

        def invoke():
            try:
                operation(*arguments)
            except BaseException as error:
                result.append(error)

        worker = threading.Thread(target=invoke, name="MainThread", daemon=True)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "worker did not finish")
        self.assertEqual(len(result), 1, "worker call was not rejected")
        self.assertTrue(
            type(result[0]) is self.module.OutputCaptureError,
            "worker rejection type mismatch",
        )
        self.assertTrue(
            result[0].args == ("tokenizer output capture invalid",),
            "worker rejection message mismatch",
        )

    def test_wrong_thread_construction_is_rejected_before_mutation(self):
        self.run_in_worker(self.module.OutputCapture)
        capture = self.module.OutputCapture()
        capture.feed("result", bytes((0, 255)))
        before = capture.snapshot()
        self.run_in_worker(self.module.OutputCapture.__init__, capture)
        self.assertTrue(capture.snapshot() == before, "worker construction mutation")

    def test_wrong_thread_methods_preserve_all_states_and_ignore_arguments(self):
        for state in ("OPEN", "FINISHED", "FAILED"):
            for method in ("feed", "end", "finish", "abort", "snapshot", "private_outputs"):
                for hostile in (False, True):
                    with self.subTest(state=state, method=method, hostile=hostile):
                        capture = self.module.OutputCapture()
                        capture.feed("result", bytes((0, 255)))
                        if state != "OPEN":
                            self.complete(capture)
                        if state == "FAILED":
                            capture.abort()
                        storage = capture._buffers["result"]
                        before = capture.snapshot()
                        if method == "feed":
                            arguments = (
                                (HostileValue(), HostileValue()) if hostile
                                else ("result", bytes((1,)))
                            )
                        elif method == "end":
                            arguments = (HostileValue() if hostile else "result",)
                        else:
                            arguments = ()
                        self.run_in_worker(getattr(capture, method), *arguments)
                        self.assertTrue(capture.snapshot() == before, "worker mutation")
                        self.assertTrue(
                            storage is capture._buffers["result"],
                            "worker storage replacement",
                        )
                        self.assertTrue(
                            storage == bytes((0, 255)), "worker storage mutation"
                        )

    def test_constructor_rejects_configuration(self):
        with self.assertRaises(TypeError):
            self.module.OutputCapture(1)
        with self.assertRaises(TypeError):
            self.module.OutputCapture(limit_bytes=1)

    def test_snapshot_is_primitive_only_in_every_state(self):
        def check(value):
            if type(value) is dict:
                self.assertTrue(
                    all(type(key) is str for key in value), "metadata key type mismatch"
                )
                for child in value.values():
                    check(child)
            elif type(value) is list:
                for child in value:
                    check(child)
            else:
                self.assertTrue(
                    type(value) in (str, int, bool, type(None)),
                    "metadata primitive type mismatch",
                )

        capture = self.module.OutputCapture()
        capture.feed("result", bytes((0, 255)))
        for state in ("OPEN", "FINISHED", "FAILED"):
            if state == "FINISHED":
                self.complete(capture)
            if state == "FAILED":
                capture.abort()
            snapshot = capture.snapshot()
            check(snapshot)
            self.assertEqual(snapshot["state"], state)
            self.assertIs(snapshot["capture_complete"], state == "FINISHED")
            self.assertEqual(snapshot["cleanup_status"], "CLEANUP_UNVERIFIED")
            for flag in FALSE_FLAGS:
                self.assertIs(snapshot[flag], False)


if __name__ == "__main__":
    unittest.main()
