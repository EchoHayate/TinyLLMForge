_LIMITS = {
    "result": 67_108_864,
    "stdout": 1_048_576,
    "stderr": 1_048_576,
}


class OutputCaptureError(ValueError):
    def __init__(self):
        super().__init__("tokenizer output capture invalid")


def _require_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise OutputCaptureError()


class OutputCapture:
    def __init__(self):
        _require_main_thread()
        self._state = "OPEN"
        self._buffers = {stream: bytearray() for stream in _LIMITS}
        self._eof = dict.fromkeys(_LIMITS, False)
        self._failure = None

    def _fail(self, reason, stream=None):
        if self._state != "FAILED":
            self._state = "FAILED"
            self._failure = {"reason": reason, "stream": stream}

    def _reject(self, reason="PROTOCOL_ERROR", stream=None):
        self._fail(reason, stream)
        raise OutputCaptureError()

    def _require_open_stream(self, stream):
        if self._state == "FAILED":
            raise OutputCaptureError()
        known = type(stream) is str and stream in _LIMITS
        if self._state != "OPEN" or not known:
            self._reject(stream=stream if known else None)
        if self._eof[stream]:
            self._reject(stream=stream)

    def feed(self, stream, chunk):
        _require_main_thread()
        self._require_open_stream(stream)
        if type(chunk) is not bytes or not 0 < len(chunk) <= 65_536:
            self._reject(stream=stream)
        if len(self._buffers[stream]) + len(chunk) > _LIMITS[stream]:
            self._reject("OUTPUT_LIMIT", stream)
        try:
            self._buffers[stream].extend(chunk)
        except BaseException as error:
            self._fail("CAPTURE_ERROR", stream)
            if isinstance(error, Exception):
                raise OutputCaptureError() from None
            raise

    def end(self, stream):
        _require_main_thread()
        self._require_open_stream(stream)
        self._eof[stream] = True

    def finish(self):
        _require_main_thread()
        if self._state == "FAILED":
            raise OutputCaptureError()
        if self._state == "FINISHED":
            return
        if not all(self._eof.values()):
            self._reject()
        self._state = "FINISHED"

    def abort(self):
        _require_main_thread()
        self._fail("ABORTED")

    def private_outputs(self):
        _require_main_thread()
        if self._state == "OPEN":
            self._reject()
        try:
            return {stream: bytes(buffer) for stream, buffer in self._buffers.items()}
        except BaseException as error:
            self._fail("CAPTURE_ERROR")
            if isinstance(error, Exception):
                raise OutputCaptureError() from None
            raise

    def snapshot(self):
        _require_main_thread()
        return {
            "kind": "LOCAL_TOKENIZER_OUTPUT_CAPTURE_ONLY",
            "state": self._state,
            "streams": [
                {
                    "stream": stream,
                    "limit_bytes": limit,
                    "accepted_bytes": len(self._buffers[stream]),
                    "eof": self._eof[stream],
                }
                for stream, limit in _LIMITS.items()
            ],
            "total_accepted_bytes": sum(map(len, self._buffers.values())),
            "capture_complete": self._state == "FINISHED",
            "failure": None if self._failure is None else dict(self._failure),
            "cleanup_status": "CLEANUP_UNVERIFIED",
            "resource_limits_enforced": False,
            "root_write_limit_enforced": False,
            "deadline_enforced": False,
            "process_termination_enforced": False,
            "cleanup_verified": False,
            "benchmark_eligible": False,
            "formal_evaluation_ready": False,
            "net_benefit_established": False,
            "isolated": False,
            "all_possible_second_rounds_validated": False,
        }
import threading


