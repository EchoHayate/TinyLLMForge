import fcntl
import importlib
import importlib.util
import os
import selectors
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from tokenizer_launch_window import LaunchWindow
from tokenizer_output_capture import OutputCapture


STREAMS = ("result", "stdout", "stderr")
WORK_NS = 540_000_000_000
SELECTOR_TYPE = selectors.DefaultSelector


class HostileValue:
    def __eq__(self, other):
        raise AssertionError("caller equality")

    def __repr__(self):
        raise AssertionError("caller formatting")


class StringSubclass(str):
    __hash__ = str.__hash__

    def __eq__(self, other):
        raise AssertionError("caller key equality")


class IntegerSubclass(int):
    pass


class DictSubclass(dict):
    def items(self):
        raise AssertionError("caller items")


class ReaderCancellation(BaseException):
    pass


class OutputReaderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tokenizer_output_reader"))
        self.module = importlib.import_module("tokenizer_output_reader")
        self.handles = set()
        self.addCleanup(self.close_handles)

    def close_handles(self):
        for descriptor in tuple(self.handles):
            os.close(descriptor)
            self.handles.remove(descriptor)

    def pipes(self, payloads=None, close_writers=True):
        streams = {}
        writers = {}
        for stream in STREAMS:
            read_fd, write_fd = os.pipe()
            self.handles.update((read_fd, write_fd))
            os.set_blocking(read_fd, False)
            streams[stream] = read_fd
            writers[stream] = write_fd
            payload = (payloads or {}).get(stream, b"")
            if payload:
                self.assertEqual(os.write(write_fd, payload), len(payload))
            if close_writers:
                os.close(write_fd)
                self.handles.remove(write_fd)
        return streams, writers

    def window(self):
        window = LaunchWindow()
        window.claim_connection()
        window.claim_audit()
        return window

    def assert_invalid(self, operation, *arguments):
        with self.assertRaises(BaseException) as caught:
            operation(*arguments)
        self.assertIs(type(caught.exception), self.module.OutputReaderError)
        self.assertEqual(caught.exception.args, ("tokenizer output reader invalid",))
        self.assertIsNone(caught.exception.__cause__)
        if caught.exception.__context__ is not None:
            self.assertTrue(caught.exception.__suppress_context__)
        return caught.exception

    def assert_failure(self, reader, reason, released=True):
        snapshot = reader.snapshot()
        self.assertEqual(snapshot["state"], "FAILED")
        self.assertEqual(snapshot["failure"], reason)
        self.assertIs(snapshot["local_handles_released"], released)
        self.assertEqual(snapshot["capture"]["state"], "FAILED")
        self.assertEqual(snapshot["capture"]["cleanup_status"], "CLEANUP_UNVERIFIED")
        flags = [value for value in snapshot["capture"].values() if type(value) is bool]
        self.assertEqual(flags, [False] * 11)

    @contextmanager
    def observe_resources(self, close_fault=None, selector_fault=None, close_before_fault=True):
        duplicates = []
        closes = []
        selector_closes = []
        real_duplicate = os.dup
        real_close = os.close
        real_selector_close = SELECTOR_TYPE.close

        def duplicate(descriptor):
            owned = real_duplicate(descriptor)
            duplicates.append(owned)
            self.handles.add(owned)
            return owned

        def close(descriptor):
            closes.append(descriptor)
            error = close_fault(len(closes)) if close_fault else None
            if error is None or close_before_fault:
                real_close(descriptor)
                self.handles.discard(descriptor)
            if error is not None:
                raise error

        def close_selector(selector):
            selector_closes.append(selector)
            real_selector_close(selector)
            if selector_fault is not None:
                raise selector_fault

        with patch("os.dup", side_effect=duplicate), patch("os.close", side_effect=close):
            with patch.object(SELECTOR_TYPE, "close", close_selector):
                yield duplicates, closes, selector_closes

    def assert_released(self, resources):
        duplicates, closes, selector_closes = resources
        self.assertEqual(closes, duplicates)
        self.assertLessEqual(len(selector_closes), 1)
        for descriptor in duplicates:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_inert_construction_and_snapshot(self):
        with patch("os.pipe") as pipe, patch("os.dup") as duplicate, patch(
            "os.read"
        ) as read, patch("selectors.DefaultSelector") as selector, patch(
            "tokenizer_launch_window.time.monotonic_ns"
        ) as clock:
            reader = self.module.OutputReader()
            original = reader.snapshot()
            changed = reader.snapshot()
            changed["capture"]["streams"][0]["accepted_bytes"] = 99
            changed["failure"] = "changed"
            self.assertEqual(reader.snapshot(), original)
        for boundary in (pipe, duplicate, read, selector, clock):
            boundary.assert_not_called()

    def test_private_outputs_before_run_does_not_mutate(self):
        reader = self.module.OutputReader()
        before = reader.snapshot()
        self.assert_invalid(reader.private_outputs)
        self.assertEqual(reader.snapshot(), before)

    def test_wrong_thread_operations_leave_reader_and_window_untouched(self):
        reader = self.module.OutputReader()
        window = self.window()
        reader_before = reader.snapshot()
        window_before = window.snapshot()
        errors = []

        def operate():
            for operation, arguments in (
                (self.module.OutputReader, ()),
                (reader.snapshot, ()),
                (reader.private_outputs, ()),
                (reader.run, (HostileValue(), HostileValue())),
            ):
                try:
                    operation(*arguments)
                except BaseException as error:
                    errors.append(error)

        worker = threading.Thread(target=operate)
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(type(error) is self.module.OutputReaderError for error in errors))
        self.assertEqual(reader.snapshot(), reader_before)
        self.assertEqual(window.snapshot(), window_before)

    def test_invalid_windows_are_not_bound_or_touched(self):
        class WindowSubclass(LaunchWindow):
            pass

        candidates = [HostileValue(), None, WindowSubclass(), LaunchWindow()]
        connecting = LaunchWindow()
        connecting.claim_connection()
        candidates.append(connecting)
        cleaning = self.window()
        cleaning.begin_cleanup("AUDIT_COMPLETE")
        candidates.append(cleaning)
        closed = self.window()
        closed.close()
        candidates.append(closed)
        for index, window in enumerate(candidates):
            with self.subTest(index=index):
                before = window.snapshot() if isinstance(window, LaunchWindow) else None
                reader = self.module.OutputReader()
                self.assert_invalid(reader.run, window, HostileValue())
                self.assert_failure(reader, "PROTOCOL_ERROR")
                if before is not None:
                    self.assertEqual(window.snapshot(), before)
                terminal = reader.snapshot()
                self.assert_invalid(reader.run, HostileValue(), HostileValue())
                self.assertEqual(reader.snapshot(), terminal)

    def test_invalid_stream_maps_fail_before_reading(self):
        streams, _ = self.pipes()
        candidates = [
            HostileValue(), DictSubclass(streams), None, {}, list(streams.items()),
            {**streams, "extra": streams["result"]},
            {StringSubclass("result"): streams["result"],
             "stdout": streams["stdout"], "stderr": streams["stderr"]},
        ]
        for value in (True, IntegerSubclass(3), -1, 0, 1, 2, 3.0, HostileValue()):
            candidates.append({**streams, "result": value})
        candidates.append({**streams, "result": streams["stdout"]})
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index):
                reader = self.module.OutputReader()
                window = self.window()
                with patch("os.read", side_effect=AssertionError("unexpected read")) as read:
                    self.assert_invalid(reader.run, window, candidate)
                read.assert_not_called()
                self.assert_failure(reader, "PROTOCOL_ERROR")
                self.assertEqual(window.snapshot()["stop_reason"], "AUDIT_FAILED")
                self.assertEqual(reader.private_outputs(), dict.fromkeys(STREAMS, b""))

    def test_disallowed_pipe_properties_and_aliases(self):
        for mode in ("blocking", "write", "file", "alias"):
            with self.subTest(mode=mode):
                streams, writers = self.pipes(dict.fromkeys(STREAMS, b"x"), close_writers=False)
                if mode == "blocking":
                    os.set_blocking(streams["stderr"], True)
                elif mode == "write":
                    os.set_blocking(writers["stderr"], False)
                    streams["stderr"] = writers["stderr"]
                elif mode == "file":
                    descriptor = os.open(__file__, os.O_RDONLY | os.O_NONBLOCK)
                    self.handles.add(descriptor)
                    streams["stderr"] = descriptor
                else:
                    duplicate = os.dup(streams["stdout"])
                    self.handles.add(duplicate)
                    streams["stderr"] = duplicate
                flags = {descriptor: fcntl.fcntl(descriptor, fcntl.F_GETFL)
                         for descriptor in streams.values()}
                reader = self.module.OutputReader()
                with patch("os.read", side_effect=AssertionError("unexpected read")) as read:
                    self.assert_invalid(reader.run, self.window(), streams)
                read.assert_not_called()
                self.assert_failure(reader, "PROTOCOL_ERROR")
                for descriptor, flag in flags.items():
                    self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GETFL), flag)

    def test_closed_descriptor_is_read_error(self):
        streams, _ = self.pipes()
        os.close(streams["stderr"])
        self.handles.remove(streams["stderr"])
        reader = self.module.OutputReader()
        with patch("os.read", side_effect=AssertionError("unexpected read")) as read:
            self.assert_invalid(reader.run, self.window(), streams)
        read.assert_not_called()
        self.assert_failure(reader, "READ_ERROR")

    def test_finished_reader_is_one_shot(self):
        streams, _ = self.pipes()
        reader = self.module.OutputReader()
        reader.run(self.window(), streams)
        before = reader.snapshot()
        self.assert_invalid(reader.run, HostileValue(), HostileValue())
        self.assertEqual(reader.snapshot(), before)

    def test_exact_deadline_checks_surround_setup_wait_read_and_finish(self):
        expected_reads = (0, 0, 0, 0, 1, 1, 2, 2, 3, 3)
        for expiry_call, read_count in enumerate(expected_reads, 1):
            with self.subTest(expiry_call=expiry_call):
                with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
                    window = self.window()
                observations = 0

                def clock():
                    nonlocal observations
                    observations += 1
                    return 100 if observations < expiry_call else WORK_NS + 100

                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                with patch("tokenizer_launch_window.time.monotonic_ns", side_effect=clock):
                    with patch("os.read", wraps=os.read) as read:
                        self.assert_invalid(reader.run, window, streams)
                self.assertEqual(read.call_count, read_count)
                self.assert_failure(reader, "WORK_DEADLINE")
                self.assertEqual(window.snapshot()["stop_reason"], "WORK_DEADLINE")
                self.assertEqual(window.snapshot()["cleanup_deadline_ns"], 600_000_000_100)

    def test_post_read_expiry_discards_only_unaccepted_chunk(self):
        streams, _ = self.pipes({"result": b"accepted", "stdout": b"discarded"})
        now = [100]
        real_read = os.read
        reads = []

        def read(descriptor, count):
            chunk = real_read(descriptor, count)
            reads.append(chunk)
            if len(reads) == 2:
                now[0] = WORK_NS + 100
            return chunk

        reader = self.module.OutputReader()
        with patch("tokenizer_launch_window.time.monotonic_ns", side_effect=lambda: now[0]):
            window = self.window()
            with patch("os.read", side_effect=read):
                self.assert_invalid(reader.run, window, streams)
        self.assertEqual(len(reads), 2)
        self.assert_failure(reader, "WORK_DEADLINE")
        self.assertEqual(reader.snapshot()["capture"]["total_accepted_bytes"], len(reads[0]))
        self.assertNotIn("discarded", str(reader.snapshot()))

    def test_wait_timeout_is_capped_by_observed_budget(self):
        for remaining in (1, 50_000_000, 200_000_000):
            with self.subTest(remaining=remaining):
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
                    window = self.window()
                real_select = selectors.DefaultSelector.select
                timeouts = []

                def select(selector, timeout):
                    timeouts.append(timeout)
                    return real_select(selector, 0)

                with patch("tokenizer_launch_window.time.monotonic_ns",
                           return_value=WORK_NS + 100 - remaining):
                    with patch.object(selectors.DefaultSelector, "select", select):
                        reader.run(window, streams)
                self.assertTrue(timeouts)
                self.assertTrue(all(0 < timeout <= min(0.1, remaining / 1_000_000_000)
                                    for timeout in timeouts))

    def test_window_cleanup_or_close_after_wait_stops_reads(self):
        for action in ("cleanup", "close"):
            with self.subTest(action=action):
                streams, _ = self.pipes()
                window = self.window()
                reader = self.module.OutputReader()
                real_select = selectors.DefaultSelector.select

                def select(selector, timeout):
                    events = real_select(selector, 0)
                    if action == "cleanup":
                        window.begin_cleanup("AUDIT_COMPLETE")
                    else:
                        window.close()
                    return events

                with patch.object(selectors.DefaultSelector, "select", select):
                    with patch("os.read", wraps=os.read) as read:
                        self.assert_invalid(reader.run, window, streams)
                read.assert_not_called()
                self.assert_failure(reader, "WORK_DEADLINE")

    def test_clock_failures_stop_before_setup(self):
        for fault in (OSError("secret clock"), -1, True, 99):
            with self.subTest(kind=type(fault).__name__):
                with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
                    window = self.window()
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                options = {"side_effect": fault} if isinstance(fault, Exception) else {
                    "return_value": fault}
                with patch("tokenizer_launch_window.time.monotonic_ns", **options):
                    with patch("os.dup", wraps=os.dup) as duplicate:
                        self.assert_invalid(reader.run, window, streams)
                duplicate.assert_not_called()
                self.assert_failure(reader, "CLOCK_ERROR")
                self.assertTrue(window.snapshot()["clock_failed"])
                self.assertNotIn("secret", str(reader.snapshot()))

    def test_transient_reads_do_not_mean_eof(self):
        for transient in (BlockingIOError(), InterruptedError()):
            with self.subTest(kind=type(transient).__name__):
                streams, _ = self.pipes({"result": b"retained"})
                reader = self.module.OutputReader()
                real_read = os.read
                real_select = selectors.DefaultSelector.select
                reads = []
                waits = []

                def select(selector, timeout):
                    waits.append(timeout)
                    return real_select(selector, 0)

                def read(descriptor, count):
                    reads.append(descriptor)
                    if len(reads) <= 3:
                        raise transient
                    return real_read(descriptor, count)

                with patch.object(selectors.DefaultSelector, "select", select):
                    with patch("os.read", side_effect=read):
                        reader.run(self.window(), streams)
                self.assertGreaterEqual(len(waits), 3)
                self.assertEqual(reader.private_outputs()["result"], b"retained")
                self.assertEqual(reader.snapshot()["state"], "FINISHED")

    def test_empty_readiness_poll_is_not_eof(self):
        streams, _ = self.pipes({"result": b"after-empty-poll"})
        reader = self.module.OutputReader()
        real_select = selectors.DefaultSelector.select
        waits = []

        def select(selector, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                return []
            return real_select(selector, 0)

        with patch.object(selectors.DefaultSelector, "select", select):
            reader.run(self.window(), streams)
        self.assertGreaterEqual(len(waits), 3)
        self.assertEqual(reader.private_outputs()["result"], b"after-empty-poll")

    def test_real_stdout_capacity_overflow_stops_reads(self):
        streams, writers = self.pipes(close_writers=False)
        for stream in ("result", "stderr"):
            os.close(writers[stream])
            self.handles.remove(writers[stream])
        write_fd = writers["stdout"]
        os.set_blocking(write_fd, False)
        stop = threading.Event()
        producer_errors = []
        produced = [0]
        limit = 1_048_576

        def produce():
            deadline = time.monotonic() + 3
            try:
                while produced[0] < limit + 1 and not stop.is_set():
                    if time.monotonic() >= deadline:
                        raise AssertionError("producer deadline")
                    chunk = b"z" * min(4096, limit + 1 - produced[0])
                    try:
                        produced[0] += os.write(write_fd, chunk)
                    except BlockingIOError:
                        stop.wait(0.001)
            except BaseException as error:
                producer_errors.append(error)
            finally:
                os.close(write_fd)

        producer = threading.Thread(target=produce)

        def stop_producer():
            stop.set()
            producer.join(4)
            self.assertFalse(producer.is_alive())
            self.assertEqual(producer_errors, [])

        producer.start()
        self.handles.remove(write_fd)
        self.addCleanup(stop_producer)
        reader = self.module.OutputReader()
        real_read = os.read
        read_sizes = []

        def read(descriptor, count):
            self.assertIsNone(reader.snapshot()["capture"]["failure"])
            read_sizes.append(count)
            return real_read(descriptor, count)

        with patch("os.read", side_effect=read):
            self.assert_invalid(reader.run, self.window(), streams)
        producer.join(4)
        self.assertFalse(producer.is_alive())
        self.assert_failure(reader, "OUTPUT_LIMIT")
        snapshot = reader.snapshot()
        self.assertEqual(snapshot["capture"]["failure"],
                         {"reason": "OUTPUT_LIMIT", "stream": "stdout"})
        self.assertLessEqual(len(reader.private_outputs()["stdout"]), limit)
        self.assertEqual(produced[0], limit + 1)
        self.assertTrue(read_sizes)
        self.assertEqual(set(read_sizes), {65_536})

    def test_setup_and_io_faults_release_partial_ownership(self):
        boundaries = (
            ("fstat", 1), ("fcntl", 2), ("selector", 1),
            ("dup", 1), ("dup", 2), ("dup", 3),
            ("register", 1), ("register", 2), ("register", 3),
            ("select", 1), ("read", 1), ("unregister", 1),
        )
        for boundary, fail_at in boundaries:
            with self.subTest(boundary=boundary, fail_at=fail_at):
                streams, _ = self.pipes()
                window = self.window()
                reader = self.module.OutputReader()
                calls = 0
                with self.observe_resources() as resources:
                    if boundary == "selector":
                        owner, attribute = selectors, "DefaultSelector"
                    elif boundary in ("register", "select", "unregister"):
                        owner, attribute = SELECTOR_TYPE, boundary
                    elif boundary == "fcntl":
                        owner, attribute = fcntl, "fcntl"
                    else:
                        owner, attribute = os, boundary
                    original = getattr(owner, attribute)

                    def fail(*arguments, **keywords):
                        nonlocal calls
                        calls += 1
                        if calls == fail_at:
                            raise OSError("secret syscall text")
                        return original(*arguments, **keywords)

                    with patch.object(owner, attribute, fail):
                        self.assert_invalid(reader.run, window, streams)
                self.assert_released(resources)
                self.assert_failure(reader, "READ_ERROR")
                self.assertEqual(window.snapshot()["stop_reason"], "AUDIT_FAILED")
                self.assertNotIn("secret", str(reader.snapshot()))
                for descriptor in streams.values():
                    os.fstat(descriptor)

    def test_capture_faults_stop_and_map_to_capture_error(self):
        for method in ("feed", "end", "finish"):
            with self.subTest(method=method):
                streams, _ = self.pipes(dict.fromkeys(STREAMS, b"payload"))
                reader = self.module.OutputReader()
                window = self.window()
                with self.observe_resources() as resources:
                    with patch.object(OutputCapture, method, side_effect=MemoryError("secret")):
                        self.assert_invalid(reader.run, window, streams)
                self.assert_failure(reader, "CAPTURE_ERROR")
                self.assert_released(resources)
                self.assertEqual(window.snapshot()["stop_reason"], "AUDIT_FAILED")

    def test_invalid_read_values_are_capture_errors_not_eof(self):
        class BytesSubclass(bytes):
            pass

        for chunk in (None, "", bytearray(), memoryview(b""), BytesSubclass(b""), b"x" * 65_537):
            with self.subTest(kind=type(chunk).__name__):
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                with patch("os.read", return_value=chunk) as read:
                    self.assert_invalid(reader.run, self.window(), streams)
                self.assertEqual(read.call_count, 1)
                self.assert_failure(reader, "CAPTURE_ERROR")
                self.assertEqual(reader.snapshot()["capture"]["failure"]["reason"], "PROTOCOL_ERROR")

    def test_release_faults_preclude_success_and_continue_all_closes(self):
        for boundary in ("selector", 1, 2, 3):
            for close_before_fault in (True, False):
                with self.subTest(boundary=boundary, closed=close_before_fault):
                    streams, _ = self.pipes()
                    reader = self.module.OutputReader()
                    window = self.window()
                    error = OSError("secret close")
                    with self.observe_resources(
                        close_fault=lambda count: error if count == boundary else None,
                        selector_fault=error if boundary == "selector" else None,
                        close_before_fault=close_before_fault,
                    ) as resources:
                        self.assert_invalid(reader.run, window, streams)
                    self.assertEqual(resources[0], resources[1])
                    self.assertEqual(len(resources[2]), 1)
                    self.assert_failure(reader, "RELEASE_ERROR", released=False)
                    self.assertEqual(window.snapshot()["stop_reason"], "AUDIT_FAILED")
                    self.assertEqual(reader.private_outputs(), dict.fromkeys(STREAMS, b""))

    def test_read_failure_survives_release_and_cleanup_errors(self):
        streams, _ = self.pipes()
        reader = self.module.OutputReader()
        with self.observe_resources(close_fault=lambda count: OSError("close")) as resources:
            with patch("os.read", side_effect=OSError("read")) as read:
                with patch.object(LaunchWindow, "begin_cleanup", side_effect=RuntimeError("cleanup")):
                    self.assert_invalid(reader.run, self.window(), streams)
        self.assertEqual(read.call_count, 1)
        self.assert_released(resources)
        self.assert_failure(reader, "READ_ERROR", released=False)

    def test_cancellation_identity_across_all_operation_boundaries(self):
        for boundary in (
            "fstat", "fcntl", "selector", "dup", "register", "select", "read",
            "unregister", "feed", "end", "finish", "clock",
        ):
            with self.subTest(boundary=boundary):
                streams, _ = self.pipes(dict.fromkeys(STREAMS, b"payload"))
                window = self.window()
                reader = self.module.OutputReader()
                cancellation = ReaderCancellation("private cancellation")
                if boundary == "selector":
                    owner, attribute = selectors, "DefaultSelector"
                elif boundary in ("register", "select", "unregister"):
                    owner, attribute = SELECTOR_TYPE, boundary
                elif boundary in ("feed", "end", "finish"):
                    owner, attribute = OutputCapture, boundary
                elif boundary == "fcntl":
                    owner, attribute = fcntl, "fcntl"
                elif boundary == "clock":
                    owner, attribute = time, "monotonic_ns"
                else:
                    owner, attribute = os, boundary
                with self.observe_resources() as resources:
                    with patch.object(owner, attribute, side_effect=cancellation):
                        with self.assertRaises(BaseException) as caught:
                            reader.run(window, streams)
                self.assertIs(caught.exception, cancellation)
                self.assert_failure(reader, "CANCELLED")
                self.assert_released(resources)
                self.assertEqual(window.snapshot()["stop_reason"],
                                 "CLOCK_ERROR" if boundary == "clock" else "CANCELLED")

    def test_earliest_cancellation_survives_later_finalization_cancellations(self):
        for primary in (None, OSError("read"), KeyboardInterrupt("first")):
            with self.subTest(primary=type(primary).__name__):
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                window = self.window()
                later = SystemExit("later")
                cleanup = ReaderCancellation("cleanup")
                with self.observe_resources(
                    selector_fault=later,
                    close_fault=lambda count: ReaderCancellation("close"),
                ) as resources:
                    with patch.object(LaunchWindow, "begin_cleanup", side_effect=cleanup):
                        with patch("os.read", side_effect=primary, wraps=os.read):
                            with self.assertRaises(BaseException) as caught:
                                reader.run(window, streams)
                expected = primary if isinstance(primary, KeyboardInterrupt) else (
                    cleanup if isinstance(primary, OSError) else later
                )
                self.assertIs(caught.exception, expected)
                reason = "READ_ERROR" if isinstance(primary, OSError) else (
                    "RELEASE_ERROR" if primary is None else "CANCELLED"
                )
                self.assert_failure(reader, reason, released=False)
                self.assert_released(resources)

    def test_abort_fault_cannot_hide_primary_failure_or_skip_cleanup(self):
        for fault in (RuntimeError("abort"), ReaderCancellation("abort")):
            with self.subTest(kind=type(fault).__name__):
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                window = self.window()
                real_abort = OutputCapture.abort

                def abort(capture):
                    real_abort(capture)
                    raise fault

                with self.observe_resources() as resources:
                    with patch.object(OutputCapture, "abort", abort):
                        with patch("os.read", side_effect=OSError("read")):
                            if isinstance(fault, Exception):
                                self.assert_invalid(reader.run, window, streams)
                            else:
                                with self.assertRaises(BaseException) as caught:
                                    reader.run(window, streams)
                                self.assertIs(caught.exception, fault)
                self.assert_failure(reader, "READ_ERROR")
                self.assert_released(resources)
                self.assertEqual(window.snapshot()["stop_reason"],
                                 "AUDIT_FAILED" if isinstance(fault, Exception) else "CANCELLED")

    def test_private_copy_failure_latches_without_overwriting_prior_failure(self):
        for failed in (False, True):
            for fault in (MemoryError("copy"), ReaderCancellation("copy")):
                with self.subTest(failed=failed, kind=type(fault).__name__):
                    streams, _ = self.pipes({"result": b"retained"})
                    reader = self.module.OutputReader()
                    window = self.window()
                    if failed:
                        with patch("os.read", side_effect=OSError("read")):
                            self.assert_invalid(reader.run, window, streams)
                    else:
                        reader.run(window, streams)
                    before = reader.snapshot()["capture"]["failure"]
                    with patch("tokenizer_output_capture.bytes", side_effect=fault, create=True):
                        if isinstance(fault, Exception):
                            self.assert_invalid(reader.private_outputs)
                        else:
                            with self.assertRaises(BaseException) as caught:
                                reader.private_outputs()
                            self.assertIs(caught.exception, fault)
                    self.assert_failure(reader, "READ_ERROR" if failed else "CAPTURE_ERROR")
                    self.assertEqual(reader.snapshot()["capture"]["failure"],
                                     before if failed else {"reason": "CAPTURE_ERROR", "stream": None})

    def test_primary_failure_aborts_and_accounts_before_release(self):
        streams, _ = self.pipes()
        reader = self.module.OutputReader()
        window = self.window()
        release_states = []
        original_close = SELECTOR_TYPE.close

        def close(selector):
            release_states.append((reader.snapshot(), window.snapshot()))
            original_close(selector)

        with patch.object(SELECTOR_TYPE, "close", close):
            with patch("os.read", side_effect=OSError("read")):
                self.assert_invalid(reader.run, window, streams)
        self.assertEqual(len(release_states), 1)
        capture_state, window_state = release_states[0]
        self.assertEqual(capture_state["capture"]["state"], "FAILED")
        self.assertEqual(window_state["stop_reason"], "AUDIT_FAILED")

    def test_binary_interleaving_feeds_before_next_read_and_unregisters_eof(self):
        payloads = {stream: bytes(range(256)) + stream.encode() for stream in STREAMS}
        streams, _ = self.pipes(payloads)
        reader = self.module.OutputReader()
        original_read = os.read
        original_select = SELECTOR_TYPE.select
        accepted = [0]
        cycle_descriptors = set()
        eof_descriptors = set()
        cycles = []

        def select(selector, timeout):
            cycles.append(tuple(cycle_descriptors))
            cycle_descriptors.clear()
            self.assertFalse(eof_descriptors.intersection(selector.get_map()))
            return original_select(selector, timeout)

        def read(descriptor, count):
            self.assertEqual(reader.snapshot()["state"], "RUNNING")
            self.assertFalse(reader.snapshot()["local_handles_released"])
            before = reader.snapshot()
            self.assert_invalid(reader.private_outputs)
            self.assertEqual(reader.snapshot(), before)
            self.assertEqual(before["capture"]["total_accepted_bytes"], accepted[0])
            self.assertNotIn(descriptor, cycle_descriptors)
            cycle_descriptors.add(descriptor)
            self.assertEqual(count, 65_536)
            chunk = original_read(descriptor, 7)
            accepted[0] += len(chunk)
            if chunk == b"":
                eof_descriptors.add(descriptor)
            return chunk

        with self.observe_resources() as resources:
            with patch.object(SELECTOR_TYPE, "select", select):
                with patch("os.read", side_effect=read):
                    reader.run(self.window(), streams)
        self.assert_released(resources)
        self.assertEqual(len(resources[0]), 3)
        self.assertEqual(len(eof_descriptors), 3)
        self.assertGreater(len(cycles), 3)
        self.assertEqual(reader.private_outputs(), payloads)

    def test_first_duplicate_close_cancellation_survives_remaining_closes(self):
        for cancellation in (KeyboardInterrupt("close"), SystemExit("close")):
            with self.subTest(kind=type(cancellation).__name__):
                streams, _ = self.pipes()
                reader = self.module.OutputReader()
                window = self.window()
                with self.observe_resources(
                    close_fault=lambda count: cancellation if count == 1 else OSError("later")
                ) as resources:
                    with self.assertRaises(BaseException) as caught:
                        reader.run(window, streams)
                self.assertIs(caught.exception, cancellation)
                self.assert_released(resources)
                self.assert_failure(reader, "RELEASE_ERROR", released=False)
                self.assertEqual(window.snapshot()["stop_reason"], "CANCELLED")

    def test_post_transient_read_deadline_stops_without_acceptance(self):
        for fault in (BlockingIOError(), InterruptedError()):
            with self.subTest(kind=type(fault).__name__):
                now = [100]
                streams, _ = self.pipes({"result": b"unaccepted"})
                reader = self.module.OutputReader()

                def read(descriptor, count):
                    now[0] = WORK_NS + 100
                    raise fault

                with patch("tokenizer_launch_window.time.monotonic_ns", side_effect=lambda: now[0]):
                    window = self.window()
                    with patch("os.read", side_effect=read) as observed_read:
                        self.assert_invalid(reader.run, window, streams)
                self.assert_failure(reader, "WORK_DEADLINE")
                self.assertEqual(observed_read.call_count, 1)
                self.assertEqual(reader.snapshot()["capture"]["total_accepted_bytes"], 0)

    def test_clock_error_after_wait_prevents_read(self):
        now = [100]
        streams, _ = self.pipes()
        reader = self.module.OutputReader()
        original_select = SELECTOR_TYPE.select

        def select(selector, timeout):
            events = original_select(selector, 0)
            now[0] = -1
            return events

        with patch("tokenizer_launch_window.time.monotonic_ns", side_effect=lambda: now[0]):
            window = self.window()
            with patch.object(SELECTOR_TYPE, "select", select):
                with patch("os.read", wraps=os.read) as read:
                    self.assert_invalid(reader.run, window, streams)
        read.assert_not_called()
        self.assert_failure(reader, "CLOCK_ERROR")

    def test_construction(self):
        reader = self.module.OutputReader()
        self.assertEqual(reader.snapshot(), {
            "kind": "LOCAL_TOKENIZER_PIPE_READER_ONLY",
            "state": "NEW",
            "failure": None,
            "local_handles_released": True,
            "capture": OutputCapture().snapshot(),
        })

    def test_real_pipes_complete_binary_and_empty_streams(self):
        payloads = {"result": b"\x00\xffprivate\n", "stdout": b"stdout\x00", "stderr": b""}
        streams, _ = self.pipes(payloads)
        flags = {stream: fcntl.fcntl(descriptor, fcntl.F_GETFL)
                 for stream, descriptor in streams.items()}
        reader = self.module.OutputReader()
        window = self.window()
        self.assertIsNone(reader.run(window, streams))
        self.assertEqual(reader.private_outputs(), payloads)
        snapshot = reader.snapshot()
        self.assertEqual(snapshot["state"], "FINISHED")
        self.assertIsNone(snapshot["failure"])
        self.assertTrue(snapshot["local_handles_released"])
        self.assertTrue(snapshot["capture"]["capture_complete"])
        self.assertEqual(window.snapshot()["state"], "AUDITING")
        for stream, descriptor in streams.items():
            os.fstat(descriptor)
            self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GETFL), flags[stream])


if __name__ == "__main__":
    unittest.main()
