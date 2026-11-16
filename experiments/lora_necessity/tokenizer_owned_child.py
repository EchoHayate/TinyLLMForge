import errno
import os
import signal
import subprocess
import threading
import time
import weakref

from tokenizer_launch_window import LaunchWindow


_HANDED_OFF = weakref.WeakSet()


class OwnedChildError(ValueError):
    def __init__(self):
        super().__init__("tokenizer owned child invalid")


def _require_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise OwnedChildError()


class OwnedChild:
    def __init__(self):
        _require_main_thread()
        self._state = "NEW"
        self._attach_attempted = False
        self._termination_attempted = False
        self._signals_attempted = []
        self._process = None
        self._pid = None
        self._exit_code = None
        self._failure = None

    def _fail(self, reason):
        self._failure = self._failure or reason

    def attach(self, process):
        _require_main_thread()
        if self._state != "NEW" or self._attach_attempted:
            raise OwnedChildError()
        self._attach_attempted = True
        try:
            if type(process) is not subprocess.Popen or process in _HANDED_OFF:
                raise OwnedChildError()
            _HANDED_OFF.add(process)
            if (
                type(process.pid) is not int
                or process.pid <= 1
                or process.pid == os.getpid()
                or process.returncode is not None
                or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
            ):
                raise OwnedChildError()
        except BaseException as error:
            self._state = "LOST"
            self._fail("PROTOCOL_ERROR" if isinstance(error, Exception) else "CANCELLED")
            if isinstance(error, Exception):
                raise OwnedChildError() from None
            raise
        self._process = process
        self._pid = process.pid
        self._state = "OWNED"
        self.poll()

    def poll(self):
        _require_main_thread()
        if self._state == "REAPED":
            return True
        if self._state != "OWNED":
            raise OwnedChildError()
        reason = "WAIT_ERROR"
        try:
            if (
                type(self._process.pid) is not int
                or self._process.pid != self._pid
                or self._process.returncode is not None
                or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
            ):
                reason = "OWNERSHIP_LOST"
                raise OwnedChildError()
            result = os.waitpid(self._pid, os.WNOHANG)
            if (
                type(result) is not tuple
                or len(result) != 2
                or any(type(value) is not int for value in result)
            ):
                raise OwnedChildError()
            returned_pid, status = result
            if returned_pid == 0 and status == 0:
                return False
            if (
                returned_pid != self._pid
                or status < 0
                or not (os.WIFEXITED(status) or os.WIFSIGNALED(status))
            ):
                raise OwnedChildError()
            exit_code = os.waitstatus_to_exitcode(status)
            self._process.returncode = exit_code
            self._exit_code = exit_code
            self._state = "REAPED"
            return True
        except BaseException as error:
            self._state = "LOST"
            if not isinstance(error, Exception):
                self._fail("CANCELLED")
                raise
            if isinstance(error, ChildProcessError) or (
                isinstance(error, OSError) and error.errno == errno.ECHILD
            ):
                reason = "OWNERSHIP_LOST"
            self._fail(reason)
            raise OwnedChildError() from None

    def _signal(self, signum):
        if self.poll():
            return True
        self._signals_attempted.append(signum.name)
        try:
            os.kill(self._pid, signum)
        except BaseException as error:
            self._fail("SIGNAL_ERROR" if isinstance(error, Exception) else "CANCELLED")
            raise
        return False

    def _budget(self, window):
        try:
            remaining = window.cleanup_remaining_ns()
            snapshot = window.snapshot()
            if snapshot["clock_failed"]:
                raise OwnedChildError()
            return remaining, snapshot
        except BaseException as error:
            self._fail("CLOCK_ERROR" if isinstance(error, Exception) else "CANCELLED")
            raise

    def _sleep(self, duration):
        try:
            time.sleep(duration)
        except BaseException as error:
            self._fail("SLEEP_ERROR" if isinstance(error, Exception) else "CANCELLED")
            raise

    def _deadline(self):
        if "SIGKILL" not in self._signals_attempted:
            self._signal(signal.SIGKILL)
        if self.poll():
            return True
        self._fail("DEADLINE")
        return False

    def _emergency(self, cancellation):
        if self._state == "OWNED" and "SIGKILL" not in self._signals_attempted:
            try:
                self._signal(signal.SIGKILL)
            except BaseException as error:
                if not isinstance(error, Exception) and cancellation is None:
                    cancellation = error
        if self._state == "OWNED":
            try:
                self.poll()
            except BaseException as error:
                if not isinstance(error, Exception) and cancellation is None:
                    cancellation = error
        return cancellation

    def terminate(self, window):
        _require_main_thread()
        if self._state == "REAPED":
            return True
        if self._state != "OWNED" or self._termination_attempted:
            raise OwnedChildError()
        try:
            if type(window) is not LaunchWindow or window.snapshot()["state"] not in (
                "CLEANUP", "CLOSED"
            ):
                raise OwnedChildError()
        except Exception:
            raise OwnedChildError() from None
        self._termination_attempted = True
        try:
            if self.poll():
                return True
            remaining, snapshot = self._budget(window)
            if remaining <= 0:
                return self._deadline()
            grace_deadline = min(
                snapshot["last_observed_ns"] + 1_000_000_000,
                snapshot["cleanup_deadline_ns"],
            )
            if self._signal(signal.SIGTERM):
                return True
            while True:
                if self.poll():
                    return True
                remaining, snapshot = self._budget(window)
                if remaining <= 0:
                    return self._deadline()
                sleep_ns = min(10_000_000, remaining)
                if "SIGKILL" not in self._signals_attempted:
                    grace_remaining = grace_deadline - snapshot["last_observed_ns"]
                    if grace_remaining <= 0:
                        if self._signal(signal.SIGKILL):
                            return True
                        continue
                    sleep_ns = min(sleep_ns, grace_remaining)
                self._sleep(sleep_ns / 1_000_000_000)
        except BaseException as error:
            cancellation = error if not isinstance(error, Exception) else None
            if cancellation is not None:
                self._fail("CANCELLED")
            cancellation = self._emergency(cancellation)
            if cancellation is not None:
                raise cancellation from None
            raise OwnedChildError() from None

    def snapshot(self):
        _require_main_thread()
        return {
            "kind": "LOCAL_OWNED_CHILD_LIFECYCLE_ONLY",
            "state": self._state,
            "attach_attempted": self._attach_attempted,
            "termination_attempted": self._termination_attempted,
            "signals_attempted": list(self._signals_attempted),
            "exit_code": self._exit_code,
            "child_reaped": self._state == "REAPED",
            "failure": self._failure,
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
