import fcntl
import os
import selectors
import stat
import threading

from tokenizer_launch_window import LaunchWindow
from tokenizer_output_capture import OutputCapture


class OutputReaderError(ValueError):
    def __init__(self):
        super().__init__("tokenizer output reader invalid")


def _require_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise OutputReaderError()


class OutputReader:
    def __init__(self):
        _require_main_thread()
        self._state = "NEW"
        self._failure = None
        self._local_handles_released = True
        self._capture = OutputCapture()
        self._window = None

    def snapshot(self):
        _require_main_thread()
        return {
            "kind": "LOCAL_TOKENIZER_PIPE_READER_ONLY",
            "state": self._state,
            "failure": self._failure,
            "local_handles_released": self._local_handles_released,
            "capture": self._capture.snapshot(),
        }

    def private_outputs(self):
        _require_main_thread()
        if self._state not in ("FINISHED", "FAILED"):
            raise OutputReaderError()
        try:
            return self._capture.private_outputs()
        except BaseException as error:
            self._fail("CAPTURE_ERROR")
            cancellation = None if isinstance(error, Exception) else error
            cancellation = self._account_failure(cancellation)
            if cancellation is not None:
                raise cancellation
            raise OutputReaderError() from None

    def _fail(self, reason):
        self._state = "FAILED"
        if self._failure is None:
            self._failure = reason

    def _reject(self, reason="PROTOCOL_ERROR"):
        self._fail(reason)
        raise OutputReaderError()

    def _account_failure(self, cancellation):
        try:
            self._capture.abort()
        except BaseException as error:
            if cancellation is None and not isinstance(error, Exception):
                cancellation = error
        if self._window is not None:
            try:
                self._window.begin_cleanup(
                    "CANCELLED" if cancellation is not None else "AUDIT_FAILED"
                )
            except BaseException as error:
                if cancellation is None and not isinstance(error, Exception):
                    cancellation = error
        return cancellation

    def _validate_streams(self, streams):
        if type(streams) is not dict:
            self._reject()
        if any(type(stream) is not str for stream in streams):
            self._reject()
        if set(streams) != {"result", "stdout", "stderr"}:
            self._reject()
        borrowed = dict(streams)
        if any(type(descriptor) is not int or descriptor < 3
               for descriptor in borrowed.values()):
            self._reject()
        if len(set(borrowed.values())) != 3:
            self._reject()
        identities = set()
        for descriptor in borrowed.values():
            metadata = os.fstat(descriptor)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            identity = (metadata.st_dev, metadata.st_ino)
            if (
                not stat.S_ISFIFO(metadata.st_mode)
                or flags & os.O_ACCMODE != os.O_RDONLY
                or not flags & os.O_NONBLOCK
                or identity in identities
            ):
                self._reject()
            identities.add(identity)
        return borrowed

    def _check_window(self, window):
        try:
            remaining = window.work_remaining_ns()
        except Exception:
            self._reject("CLOCK_ERROR")
        snapshot = window.snapshot()
        if snapshot["clock_failed"]:
            self._reject("CLOCK_ERROR")
        if snapshot["state"] != "AUDITING" or remaining <= 0:
            self._reject("WORK_DEADLINE")
        return remaining

    def run(self, window, streams):
        _require_main_thread()
        if self._state != "NEW":
            raise OutputReaderError()
        self._state = "RUNNING"
        owned = []
        selector = None
        cancellation = None
        accounted = False
        phase = "READ_ERROR"
        try:
            if type(window) is not LaunchWindow or window.snapshot()["state"] != "AUDITING":
                self._reject()
            self._window = window
            borrowed = self._validate_streams(streams)
            self._check_window(window)
            selector = selectors.DefaultSelector()
            self._local_handles_released = False
            for stream, descriptor in borrowed.items():
                duplicate = os.dup(descriptor)
                owned.append(duplicate)
                selector.register(duplicate, selectors.EVENT_READ, stream)
            while selector.get_map():
                remaining = self._check_window(window)
                events = selector.select(min(0.1, remaining / 1_000_000_000))
                self._check_window(window)
                for key, _ in events:
                    self._check_window(window)
                    try:
                        chunk = os.read(key.fd, 65_536)
                    except (BlockingIOError, InterruptedError):
                        self._check_window(window)
                        continue
                    self._check_window(window)
                    phase = "CAPTURE_ERROR"
                    if type(chunk) is bytes and chunk == b"":
                        self._capture.end(key.data)
                        phase = "READ_ERROR"
                        selector.unregister(key.fd)
                    else:
                        self._capture.feed(key.data, chunk)
                    phase = "READ_ERROR"
            self._check_window(window)
            phase = "CAPTURE_ERROR"
            self._capture.finish()
        except BaseException as error:
            if not isinstance(error, Exception):
                cancellation = error
                phase = "CANCELLED"
            elif phase == "CAPTURE_ERROR":
                capture_failure = self._capture.snapshot()["failure"]
                if capture_failure is not None and capture_failure["reason"] == "OUTPUT_LIMIT":
                    phase = "OUTPUT_LIMIT"
            self._fail(phase)
            cancellation = self._account_failure(cancellation)
            accounted = True
        finally:
            released = True
            if selector is not None:
                try:
                    selector.close()
                except BaseException as error:
                    released = False
                    self._fail("RELEASE_ERROR")
                    if cancellation is None and not isinstance(error, Exception):
                        cancellation = error
            for duplicate in owned:
                try:
                    os.close(duplicate)
                except BaseException as error:
                    released = False
                    self._fail("RELEASE_ERROR")
                    if cancellation is None and not isinstance(error, Exception):
                        cancellation = error
            self._local_handles_released = released
        if self._failure is not None:
            if not accounted:
                cancellation = self._account_failure(cancellation)
            if cancellation is not None:
                raise cancellation
            raise OutputReaderError() from None
        self._state = "FINISHED"
