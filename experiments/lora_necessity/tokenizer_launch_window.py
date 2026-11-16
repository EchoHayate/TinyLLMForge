import threading
import time


class LaunchWindowError(ValueError):
    def __init__(self):
        super().__init__("tokenizer launch window invalid")


def _require_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise LaunchWindowError()


class LaunchWindow:
    def __init__(self, *, work_timeout_seconds=540):
        _require_main_thread()
        if type(work_timeout_seconds) is not int or not 0 < work_timeout_seconds <= 1200:
            raise LaunchWindowError()
        self._work_budget_ns = work_timeout_seconds * 1_000_000_000
        self._state = "NEW"
        self._connection_attempts = 0
        self._audit_attempts = 0
        self._started_ns = None
        self._last_observed_ns = None
        self._work_deadline_ns = None
        self._total_deadline_ns = None
        self._cleanup_deadline_ns = None
        self._stop_reason = None
        self._clock_failed = False

    def _enter_cleanup(self, reason, deadline):
        if self._state in ("CONNECTING", "AUDITING"):
            self._state = "CLEANUP"
            self._stop_reason = self._stop_reason or reason
            self._cleanup_deadline_ns = deadline

    def _observe(self):
        try:
            observed = time.monotonic_ns()
            if (
                type(observed) is not int
                or observed < 0
                or (
                    self._last_observed_ns is not None
                    and observed < self._last_observed_ns
                )
            ):
                raise LaunchWindowError()
        except BaseException as error:
            self._clock_failed = True
            self._state = "CLOSED"
            self._stop_reason = self._stop_reason or "CLOCK_ERROR"
            if isinstance(error, Exception):
                raise LaunchWindowError() from None
            raise
        self._last_observed_ns = observed
        if self._state in ("CONNECTING", "AUDITING") and observed >= self._work_deadline_ns:
            self._enter_cleanup("WORK_DEADLINE", self._total_deadline_ns)
        if self._state == "CLEANUP" and observed >= self._cleanup_deadline_ns:
            self._state = "CLOSED"
        return observed

    def _protocol_error(self):
        if self._state == "NEW":
            self._state = "CLOSED"
            self._stop_reason = "PROTOCOL_ERROR"
        elif self._state != "CLOSED":
            observed = self._observe()
            self._enter_cleanup(
                "PROTOCOL_ERROR", min(observed + 60_000_000_000, self._total_deadline_ns)
            )
        raise LaunchWindowError()

    def claim_connection(self):
        _require_main_thread()
        if self._state != "NEW":
            self._protocol_error()
        self._connection_attempts = 1
        self._started_ns = self._observe()
        self._work_deadline_ns = self._started_ns + self._work_budget_ns
        self._total_deadline_ns = self._work_deadline_ns + 60_000_000_000
        self._state = "CONNECTING"

    def claim_audit(self):
        _require_main_thread()
        if self._state != "CONNECTING":
            self._protocol_error()
        self._observe()
        if self._state != "CONNECTING":
            raise LaunchWindowError()
        self._audit_attempts = 1
        self._state = "AUDITING"

    def work_remaining_ns(self):
        _require_main_thread()
        if self._state == "NEW":
            self._protocol_error()
        if self._state == "CLOSED":
            return 0
        observed = self._observe()
        if self._state in ("CONNECTING", "AUDITING"):
            return max(self._work_deadline_ns - observed, 0)
        return 0

    def begin_cleanup(self, reason):
        _require_main_thread()
        if type(reason) is not str or reason not in (
            "AUDIT_COMPLETE", "AUDIT_FAILED", "CONNECTION_FAILED", "CANCELLED"
        ):
            self._protocol_error()
        if self._state == "NEW":
            self._state = "CLOSED"
            self._stop_reason = reason
        elif self._state != "CLOSED":
            observed = self._observe()
            self._enter_cleanup(
                reason, min(observed + 60_000_000_000, self._total_deadline_ns)
            )

    def cleanup_remaining_ns(self):
        _require_main_thread()
        if self._state not in ("CLEANUP", "CLOSED"):
            self._protocol_error()
        if self._state == "CLOSED":
            return 0
        observed = self._observe()
        if self._state == "CLOSED":
            return 0
        return max(self._cleanup_deadline_ns - observed, 0)

    def close(self):
        _require_main_thread()
        if self._state == "CLOSED":
            return
        if self._state != "NEW":
            self._observe()
        self._state = "CLOSED"
        self._stop_reason = self._stop_reason or "CLOSED"

    def snapshot(self):
        _require_main_thread()
        return {
            "kind": "LOCAL_TOKENIZER_LAUNCH_WINDOW_ACCOUNTING_ONLY",
            "state": self._state,
            "connection_attempts": self._connection_attempts,
            "audit_attempts": self._audit_attempts,
            "started_ns": self._started_ns,
            "last_observed_ns": self._last_observed_ns,
            "work_deadline_ns": self._work_deadline_ns,
            "total_deadline_ns": self._total_deadline_ns,
            "cleanup_deadline_ns": self._cleanup_deadline_ns,
            "stop_reason": self._stop_reason,
            "clock_failed": self._clock_failed,
            "cleanup_status": "CLEANUP_UNVERIFIED",
            "resource_limits_enforced": False,
            "process_termination_enforced": False,
            "cleanup_verified": False,
            "benchmark_eligible": False,
            "formal_evaluation_ready": False,
            "net_benefit_established": False,
            "isolated": False,
            "all_possible_second_rounds_validated": False,
        }
