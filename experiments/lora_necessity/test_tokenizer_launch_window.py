import importlib
import importlib.util
import threading
import unittest
from unittest.mock import patch


FALSE_FLAGS = (
    "resource_limits_enforced",
    "process_termination_enforced",
    "cleanup_verified",
    "benchmark_eligible",
    "formal_evaluation_ready",
    "net_benefit_established",
    "isolated",
    "all_possible_second_rounds_validated",
)
WORK_NS = 540_000_000_000
TOTAL_NS = 600_000_000_000
CLEANUP_NS = 60_000_000_000
REASONS = ("AUDIT_COMPLETE", "AUDIT_FAILED", "CONNECTION_FAILED", "CANCELLED")


class StringSubclass(str):
    pass


class IntegerSubclass(int):
    pass


class UntrustedReason:
    def __eq__(self, other):
        raise AssertionError("caller equality must not run")

    def __str__(self):
        raise AssertionError("caller formatting must not run")


class LaunchWindowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("tokenizer_launch_window"),
            "LaunchWindow implementation is missing",
        )
        self.module = importlib.import_module("tokenizer_launch_window")

    def test_construction_and_snapshot_do_not_sample_time(self):
        with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
            window = self.module.LaunchWindow()
            expected = {
                "kind": "LOCAL_TOKENIZER_LAUNCH_WINDOW_ACCOUNTING_ONLY",
                "state": "NEW",
                "connection_attempts": 0,
                "audit_attempts": 0,
                "started_ns": None,
                "last_observed_ns": None,
                "work_deadline_ns": None,
                "total_deadline_ns": None,
                "cleanup_deadline_ns": None,
                "stop_reason": None,
                "clock_failed": False,
                "cleanup_status": "CLEANUP_UNVERIFIED",
                **dict.fromkeys(FALSE_FLAGS, False),
            }
            self.assertEqual(window.snapshot(), expected)
            clock.assert_not_called()

    def test_connection_fixes_absolute_deadlines(self):
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
            window = self.module.LaunchWindow()
            self.assertIsNone(window.claim_connection())
            self.assertEqual(window.work_remaining_ns(), WORK_NS)
            self.assertEqual(window.snapshot()["total_deadline_ns"], 600_000_000_100)
            self.assertEqual(window.snapshot()["started_ns"], 100)
            self.assertEqual(window.snapshot()["work_deadline_ns"], WORK_NS + 100)
            self.assertEqual(window.snapshot()["connection_attempts"], 1)
            self.assertEqual(window.snapshot()["state"], "CONNECTING")

    def test_explicit_work_timeout_is_absolute_and_bounded(self):
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100) as clock:
            window = self.module.LaunchWindow(work_timeout_seconds=1200)
            window.claim_connection()
            self.assertEqual(window.work_remaining_ns(), 1_200_000_000_000)
            self.assertEqual(window.snapshot()["total_deadline_ns"], 1_260_000_000_100)
            clock.return_value = 540_000_000_100
            window.claim_audit()
            self.assertEqual(window.work_remaining_ns(), 660_000_000_000)
            clock.return_value = 1_200_000_000_100
            self.assertEqual(window.work_remaining_ns(), 0)
            self.assertEqual(window.snapshot()["state"], "CLEANUP")
            self.assertEqual(window.cleanup_remaining_ns(), CLEANUP_NS)
        for invalid in (0, -1, 1201, True, 1200.0, "1200", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.module.LaunchWindow(work_timeout_seconds=invalid)

    def make_window(self, state="CONNECTING", started=100):
        for method in (
            "claim_connection", "claim_audit", "work_remaining_ns",
            "begin_cleanup", "cleanup_remaining_ns", "close", "snapshot",
        ):
            self.assertTrue(
                callable(getattr(self.module.LaunchWindow, method, None)),
                f"LaunchWindow.{method} is missing",
            )
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=started):
            window = self.module.LaunchWindow()
            if state != "NEW":
                window.claim_connection()
            if state == "AUDITING":
                window.claim_audit()
            if state == "CLEANUP":
                window.begin_cleanup("CANCELLED")
            if state == "CLOSED":
                window.close()
        self.assertEqual(window.snapshot()["state"], state)
        return window

    def assert_invalid(self, operation, *arguments):
        self.assertTrue(
            hasattr(self.module, "LaunchWindowError"), "LaunchWindowError is missing"
        )
        with self.assertRaises(BaseException) as caught:
            operation(*arguments)
        self.assertIsInstance(caught.exception, self.module.LaunchWindowError)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(str(caught.exception), "tokenizer launch window invalid")
        self.assertEqual(caught.exception.args, ("tokenizer launch window invalid",))
        return caught.exception

    def test_audit_claim_does_not_reset_work_or_total_deadline(self):
        window = self.make_window()
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=123) as clock:
            self.assertIsNone(window.claim_audit())
            clock.assert_called_once_with()
            self.assertEqual(window.work_remaining_ns(), WORK_NS - 23)
        snapshot = window.snapshot()
        self.assertEqual(snapshot["state"], "AUDITING")
        self.assertEqual(snapshot["connection_attempts"], 1)
        self.assertEqual(snapshot["audit_attempts"], 1)
        self.assertEqual(snapshot["work_deadline_ns"], 100 + WORK_NS)
        self.assertEqual(snapshot["total_deadline_ns"], 100 + TOTAL_NS)

    def test_work_boundary_uses_reserved_final_minute(self):
        for state in ("CONNECTING", "AUDITING"):
            for offset in (-1, 0, 1):
                with self.subTest(state=state, offset=offset):
                    window = self.make_window(state)
                    observed = 100 + WORK_NS + offset
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=observed
                    ) as clock:
                        self.assertEqual(window.work_remaining_ns(), max(-offset, 0))
                        clock.assert_called_once_with()
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["last_observed_ns"], observed)
                    self.assertEqual(snapshot["state"], state if offset < 0 else "CLEANUP")
                    self.assertEqual(
                        snapshot["stop_reason"], None if offset < 0 else "WORK_DEADLINE"
                    )
                    self.assertEqual(
                        snapshot["cleanup_deadline_ns"], None if offset < 0 else 100 + TOTAL_NS
                    )

    def test_audit_claim_at_work_boundary_never_consumes_expired_claim(self):
        for offset in (-1, 0, 1, CLEANUP_NS, CLEANUP_NS + 1):
            with self.subTest(offset=offset):
                window = self.make_window()
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns",
                    return_value=100 + WORK_NS + offset,
                ) as clock:
                    if offset < 0:
                        self.assertIsNone(window.claim_audit())
                    else:
                        self.assert_invalid(window.claim_audit)
                    clock.assert_called_once_with()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["audit_attempts"], int(offset < 0))
                expected = "AUDITING" if offset < 0 else (
                    "CLEANUP" if offset < CLEANUP_NS else "CLOSED"
                )
                self.assertEqual(snapshot["state"], expected)
                self.assertEqual(
                    snapshot["stop_reason"], None if offset < 0 else "WORK_DEADLINE"
                )

    def test_total_boundary_closes_without_negative_work_time(self):
        for state in ("CONNECTING", "AUDITING"):
            for offset in (-1, 0, 1):
                with self.subTest(state=state, offset=offset):
                    window = self.make_window(state)
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns",
                        return_value=100 + TOTAL_NS + offset,
                    ) as clock:
                        self.assertEqual(window.work_remaining_ns(), 0)
                        clock.assert_called_once_with()
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["state"], "CLEANUP" if offset < 0 else "CLOSED")
                    self.assertEqual(snapshot["cleanup_deadline_ns"], 100 + TOTAL_NS)
                    self.assertEqual(snapshot["stop_reason"], "WORK_DEADLINE")

    def test_early_cleanup_ends_before_total_and_never_resets(self):
        for state in ("CONNECTING", "AUDITING"):
            for reason in REASONS:
                with self.subTest(state=state, reason=reason):
                    window = self.make_window(state)
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=200
                    ) as clock:
                        self.assertIsNone(window.begin_cleanup(reason))
                        clock.assert_called_once_with()
                    deadline = 200 + CLEANUP_NS
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=201
                    ) as clock:
                        self.assertIsNone(window.begin_cleanup("AUDIT_FAILED"))
                        clock.assert_called_once_with()
                        self.assertEqual(window.work_remaining_ns(), 0)
                        self.assertEqual(window.cleanup_remaining_ns(), CLEANUP_NS - 1)
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["cleanup_deadline_ns"], deadline)
                    self.assertLess(deadline, snapshot["total_deadline_ns"])
                    self.assertEqual(snapshot["stop_reason"], reason)

    def test_cleanup_boundary_closes_exactly_at_early_deadline(self):
        for offset in (-1, 0, 1):
            with self.subTest(offset=offset):
                window = self.make_window("CLEANUP")
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns",
                    return_value=100 + CLEANUP_NS + offset,
                ) as clock:
                    self.assertEqual(window.cleanup_remaining_ns(), max(-offset, 0))
                    clock.assert_called_once_with()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["state"], "CLEANUP" if offset < 0 else "CLOSED")
                self.assertEqual(snapshot["stop_reason"], "CANCELLED")

    def test_cleanup_observation_methods_all_honor_early_deadline(self):
        for method, arguments, invalid in (
            ("work_remaining_ns", (), False),
            ("cleanup_remaining_ns", (), False),
            ("begin_cleanup", ("AUDIT_COMPLETE",), False),
            ("close", (), False),
            ("claim_connection", (), True),
            ("claim_audit", (), True),
            ("begin_cleanup", ("private payload",), True),
        ):
            with self.subTest(method=method, arguments=arguments):
                window = self.make_window("CLEANUP")
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns",
                    return_value=100 + CLEANUP_NS,
                ) as clock:
                    operation = getattr(window, method)
                    if invalid:
                        self.assert_invalid(operation, *arguments)
                    else:
                        operation(*arguments)
                    clock.assert_called_once_with()
                self.assertEqual(window.snapshot()["state"], "CLOSED")
                self.assertEqual(window.snapshot()["stop_reason"], "CANCELLED")

    def test_cleanup_at_work_expiry_preserves_deadline_precedence(self):
        for offset in (-1, 0, 1, CLEANUP_NS, CLEANUP_NS + 1):
            with self.subTest(offset=offset):
                window = self.make_window()
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns",
                    return_value=100 + WORK_NS + offset,
                ) as clock:
                    self.assertIsNone(window.begin_cleanup("AUDIT_COMPLETE"))
                    clock.assert_called_once_with()
                snapshot = window.snapshot()
                self.assertEqual(
                    snapshot["stop_reason"],
                    "AUDIT_COMPLETE" if offset < 0 else "WORK_DEADLINE",
                )
                self.assertEqual(
                    snapshot["cleanup_deadline_ns"], 100 + TOTAL_NS + min(offset, 0)
                )
                self.assertEqual(
                    snapshot["state"], "CLEANUP" if offset < CLEANUP_NS else "CLOSED"
                )

    def test_cleanup_in_new_closes_without_clock_or_deadlines(self):
        for reason in REASONS:
            with self.subTest(reason=reason):
                window = self.make_window("NEW")
                with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
                    self.assertIsNone(window.begin_cleanup(reason))
                    clock.assert_not_called()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["state"], "CLOSED")
                self.assertEqual(snapshot["stop_reason"], reason)
                for field in ("started_ns", "work_deadline_ns", "total_deadline_ns",
                              "last_observed_ns", "cleanup_deadline_ns"):
                    self.assertIsNone(snapshot[field])

    def test_close_in_every_state_preserves_first_reason(self):
        for state in ("NEW", "CONNECTING", "AUDITING", "CLEANUP", "CLOSED"):
            with self.subTest(state=state):
                window = self.make_window(state)
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=101
                ) as clock:
                    self.assertIsNone(window.close())
                    self.assertEqual(clock.call_count, int(state not in ("NEW", "CLOSED")))
                self.assertEqual(window.snapshot()["state"], "CLOSED")
                self.assertEqual(
                    window.snapshot()["stop_reason"], "CANCELLED" if state == "CLEANUP" else "CLOSED"
                )

    def test_close_observes_work_expiry_before_recording_reason(self):
        for observed in (100 + WORK_NS, 100 + TOTAL_NS + 1):
            with self.subTest(observed=observed):
                window = self.make_window()
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=observed
                ) as clock:
                    self.assertIsNone(window.close())
                    clock.assert_called_once_with()
                self.assertEqual(window.snapshot()["state"], "CLOSED")
                self.assertEqual(window.snapshot()["stop_reason"], "WORK_DEADLINE")
                self.assertEqual(window.snapshot()["cleanup_deadline_ns"], 100 + TOTAL_NS)

    def test_protocol_error_matrix_preserves_claim_counts(self):
        operations = {
            "NEW": ("claim_audit", "work_remaining_ns", "cleanup_remaining_ns"),
            "CONNECTING": ("claim_connection", "cleanup_remaining_ns"),
            "AUDITING": ("claim_connection", "claim_audit", "cleanup_remaining_ns"),
            "CLEANUP": ("claim_connection", "claim_audit"),
            "CLOSED": ("claim_connection", "claim_audit"),
        }
        for state, methods in operations.items():
            for method in methods:
                with self.subTest(state=state, method=method):
                    window = self.make_window(state)
                    before = window.snapshot()
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=101
                    ) as clock:
                        self.assert_invalid(getattr(window, method))
                        self.assertEqual(clock.call_count, int(state not in ("NEW", "CLOSED")))
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["connection_attempts"], before["connection_attempts"])
                    self.assertEqual(snapshot["audit_attempts"], before["audit_attempts"])
                    self.assertEqual(
                        snapshot["state"], "CLOSED" if state in ("NEW", "CLOSED") else "CLEANUP"
                    )
                    self.assertEqual(
                        snapshot["stop_reason"], before["stop_reason"] or "PROTOCOL_ERROR"
                    )
                    expected_deadline = (
                        101 + CLEANUP_NS if state in ("CONNECTING", "AUDITING")
                        else before["cleanup_deadline_ns"]
                    )
                    self.assertEqual(snapshot["cleanup_deadline_ns"], expected_deadline)
                    if state == "NEW":
                        self.assertIsNone(snapshot["last_observed_ns"])
                    if state == "CLOSED":
                        self.assertEqual(snapshot, before)

    def test_protocol_error_at_expiry_uses_work_deadline_reason(self):
        for state, method in (
            ("CONNECTING", "claim_connection"),
            ("CONNECTING", "cleanup_remaining_ns"),
            ("AUDITING", "claim_audit"),
            ("AUDITING", "claim_connection"),
        ):
            for offset in (0, 1, CLEANUP_NS, CLEANUP_NS + 1):
                with self.subTest(state=state, method=method, offset=offset):
                    window = self.make_window(state)
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns",
                        return_value=100 + WORK_NS + offset,
                    ) as clock:
                        self.assert_invalid(getattr(window, method))
                        clock.assert_called_once_with()
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["stop_reason"], "WORK_DEADLINE")
                    self.assertEqual(snapshot["cleanup_deadline_ns"], 100 + TOTAL_NS)
                    self.assertEqual(
                        snapshot["state"], "CLEANUP" if offset < CLEANUP_NS else "CLOSED"
                    )

    def test_invalid_cleanup_reasons_are_protocol_errors_in_every_state(self):
        for state in ("NEW", "CONNECTING", "AUDITING", "CLEANUP", "CLOSED"):
            for reason in (
                None, True, 1, [], {}, b"CANCELLED", "", "private payload",
                "CLOCK_ERROR", "cancelled", StringSubclass("CANCELLED"), UntrustedReason(),
            ):
                with self.subTest(state=state, reason_type=type(reason).__name__):
                    window = self.make_window(state)
                    before = window.snapshot()
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=101
                    ) as clock:
                        self.assert_invalid(window.begin_cleanup, reason)
                        self.assertEqual(clock.call_count, int(state not in ("NEW", "CLOSED")))
                    snapshot = window.snapshot()
                    self.assertEqual(
                        snapshot["stop_reason"], before["stop_reason"] or "PROTOCOL_ERROR"
                    )
                    self.assertEqual(
                        snapshot["state"], "CLOSED" if state in ("NEW", "CLOSED") else "CLEANUP"
                    )
                    self.assertNotIn("private payload", str(snapshot))
                    self.assertEqual(snapshot["connection_attempts"], before["connection_attempts"])
                    self.assertEqual(snapshot["audit_attempts"], before["audit_attempts"])

    def test_protocol_failure_retains_usable_bounded_cleanup(self):
        window = self.make_window()
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=200):
            self.assert_invalid(window.claim_connection)
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=201):
            self.assert_invalid(window.claim_audit)
            self.assertIsNone(window.begin_cleanup("CANCELLED"))
            self.assertEqual(window.cleanup_remaining_ns(), CLEANUP_NS - 1)
        self.assertEqual(window.snapshot()["cleanup_deadline_ns"], 200 + CLEANUP_NS)
        self.assertEqual(window.snapshot()["stop_reason"], "PROTOCOL_ERROR")

    def test_closed_is_terminal_and_never_samples(self):
        window = self.make_window("CLOSED")
        before = window.snapshot()
        with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
            self.assertIsNone(window.close())
            self.assertIsNone(window.begin_cleanup("CANCELLED"))
            self.assertEqual(window.work_remaining_ns(), 0)
            self.assertEqual(window.cleanup_remaining_ns(), 0)
            self.assert_invalid(window.claim_connection)
            self.assert_invalid(window.claim_audit)
            self.assert_invalid(window.begin_cleanup, "invalid")
            self.assertEqual(window.snapshot(), before)
            clock.assert_not_called()

    def test_snapshot_is_independent_primitive_only_and_last_observed(self):
        for state in ("NEW", "CONNECTING", "AUDITING", "CLEANUP", "CLOSED"):
            with self.subTest(state=state):
                window = self.make_window(state)
                with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
                    snapshot = window.snapshot()
                    original = dict(snapshot)
                    snapshot.clear()
                    snapshot["benchmark_eligible"] = True
                    snapshot["state"] = "NEW"
                    self.assertEqual(window.snapshot(), original)
                    self.assertIsNot(window.snapshot(), window.snapshot())
                    clock.assert_not_called()
                for value in original.values():
                    self.assertIn(type(value), (str, int, bool, type(None)))
                for flag in FALSE_FLAGS:
                    self.assertIs(original[flag], False)
                self.assertEqual(original["cleanup_status"], "CLEANUP_UNVERIFIED")
                self.assertEqual(
                    original["kind"], "LOCAL_TOKENIZER_LAUNCH_WINDOW_ACCOUNTING_ONLY"
                )
                self.assertFalse(any("remaining" in key for key in original))

    def test_zero_equal_and_arbitrarily_large_integer_clocks(self):
        for started in (0, 100, 10 ** 100):
            with self.subTest(started=started):
                window = self.make_window(started=started)
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=started
                ) as clock:
                    self.assertIsNone(window.claim_audit())
                    self.assertEqual(window.work_remaining_ns(), WORK_NS)
                    self.assertIsNone(window.begin_cleanup("AUDIT_COMPLETE"))
                    self.assertEqual(window.cleanup_remaining_ns(), CLEANUP_NS)
                    self.assertEqual(clock.call_count, 4)
                snapshot = window.snapshot()
                self.assertEqual(snapshot["last_observed_ns"], started)
                self.assertEqual(snapshot["work_deadline_ns"], started + WORK_NS)
                self.assertEqual(snapshot["total_deadline_ns"], started + TOTAL_NS)
                self.assertIs(snapshot["clock_failed"], False)

    def test_invalid_initial_clock_consumes_connection_without_deadlines(self):
        for sample in (True, False, IntegerSubclass(100), -1, 100.0, None, "100", []):
            with self.subTest(sample_type=type(sample).__name__):
                window = self.make_window("NEW")
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=sample
                ) as clock:
                    self.assert_invalid(window.claim_connection)
                    clock.assert_called_once_with()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["connection_attempts"], 1)
                self.assertEqual(snapshot["audit_attempts"], 0)
                self.assertEqual(snapshot["state"], "CLOSED")
                self.assertEqual(snapshot["stop_reason"], "CLOCK_ERROR")
                self.assertIs(snapshot["clock_failed"], True)
                for field in ("started_ns", "last_observed_ns", "work_deadline_ns",
                              "total_deadline_ns", "cleanup_deadline_ns"):
                    self.assertIsNone(snapshot[field])
                with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
                    self.assert_invalid(window.claim_connection)
                    self.assertEqual(window.work_remaining_ns(), 0)
                    self.assertEqual(window.cleanup_remaining_ns(), 0)
                    self.assertIsNone(window.begin_cleanup("CANCELLED"))
                    self.assertEqual(window.snapshot(), snapshot)
                    clock.assert_not_called()

    def test_clock_faults_at_every_timed_entry_latch_closed_and_keep_valid_sample(self):
        operations = (
            ("NEW", "claim_connection", ()),
            ("CONNECTING", "claim_audit", ()),
            ("CONNECTING", "work_remaining_ns", ()),
            ("CONNECTING", "begin_cleanup", ("AUDIT_COMPLETE",)),
            ("CONNECTING", "close", ()),
            ("CONNECTING", "claim_connection", ()),
            ("CONNECTING", "cleanup_remaining_ns", ()),
            ("CONNECTING", "begin_cleanup", ("private payload",)),
            ("AUDITING", "work_remaining_ns", ()),
            ("AUDITING", "begin_cleanup", ("AUDIT_FAILED",)),
            ("AUDITING", "close", ()),
            ("AUDITING", "claim_audit", ()),
            ("AUDITING", "claim_connection", ()),
            ("AUDITING", "cleanup_remaining_ns", ()),
            ("CLEANUP", "work_remaining_ns", ()),
            ("CLEANUP", "cleanup_remaining_ns", ()),
            ("CLEANUP", "begin_cleanup", ("AUDIT_COMPLETE",)),
            ("CLEANUP", "begin_cleanup", ("private payload",)),
            ("CLEANUP", "close", ()),
            ("CLEANUP", "claim_connection", ()),
            ("CLEANUP", "claim_audit", ()),
        )
        for state, method, arguments in operations:
            for fault in (
                RuntimeError("private clock payload"),
                ValueError("private clock payload"),
                True, IntegerSubclass(100), -1, 1.0, None, "100", [],
            ):
                with self.subTest(state=state, method=method, fault_type=type(fault).__name__):
                    window = self.make_window(state)
                    before = window.snapshot()
                    clock_options = (
                        {"side_effect": fault} if isinstance(fault, Exception)
                        else {"return_value": fault}
                    )
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", **clock_options
                    ) as clock:
                        failure = self.assert_invalid(getattr(window, method), *arguments)
                        clock.assert_called_once_with()
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["state"], "CLOSED")
                    self.assertIs(snapshot["clock_failed"], True)
                    self.assertEqual(
                        snapshot["stop_reason"], before["stop_reason"] or "CLOCK_ERROR"
                    )
                    self.assertEqual(snapshot["last_observed_ns"], before["last_observed_ns"])
                    self.assertEqual(snapshot["audit_attempts"], before["audit_attempts"])
                    self.assertEqual(snapshot["connection_attempts"], 1)
                    for field in ("started_ns", "work_deadline_ns", "total_deadline_ns",
                                  "cleanup_deadline_ns"):
                        self.assertEqual(snapshot[field], before[field])
                    self.assertNotIn("private", str(failure))
                    self.assertIsNone(failure.__cause__)
                    if isinstance(fault, Exception):
                        self.assertTrue(failure.__suppress_context__)
                    for flag in FALSE_FLAGS:
                        self.assertIs(snapshot[flag], False)

    def test_clock_regression_preserves_latest_valid_observation(self):
        for state in ("CONNECTING", "AUDITING", "CLEANUP"):
            with self.subTest(state=state):
                window = self.make_window(state)
                with patch("tokenizer_launch_window.time.monotonic_ns", return_value=200):
                    window.work_remaining_ns()
                before = window.snapshot()
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=199
                ) as clock:
                    self.assert_invalid(window.work_remaining_ns)
                    clock.assert_called_once_with()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["last_observed_ns"], 200)
                self.assertEqual(snapshot["state"], "CLOSED")
                self.assertEqual(snapshot["stop_reason"], before["stop_reason"] or "CLOCK_ERROR")
                self.assertIs(snapshot["clock_failed"], True)
                self.assertEqual(snapshot["cleanup_deadline_ns"], before["cleanup_deadline_ns"])

    def test_clock_baseexceptions_rethrow_original_objects_after_latching(self):
        for state, method, arguments in (
            ("NEW", "claim_connection", ()),
            ("CONNECTING", "claim_audit", ()),
            ("CONNECTING", "work_remaining_ns", ()),
            ("CONNECTING", "begin_cleanup", ("CANCELLED",)),
            ("CONNECTING", "claim_connection", ()),
            ("CONNECTING", "begin_cleanup", (None,)),
            ("AUDITING", "close", ()),
            ("CLEANUP", "cleanup_remaining_ns", ()),
            ("CLEANUP", "close", ()),
        ):
            for exception_type in (KeyboardInterrupt, SystemExit, GeneratorExit):
                with self.subTest(state=state, method=method, fault=exception_type.__name__):
                    window = self.make_window(state)
                    before = window.snapshot()
                    original = exception_type("private cancellation payload")
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", side_effect=original
                    ) as clock:
                        with self.assertRaises(BaseException) as caught:
                            getattr(window, method)(*arguments)
                        clock.assert_called_once_with()
                    self.assertIs(caught.exception, original)
                    snapshot = window.snapshot()
                    self.assertEqual(snapshot["state"], "CLOSED")
                    self.assertIs(snapshot["clock_failed"], True)
                    self.assertEqual(
                        snapshot["stop_reason"], before["stop_reason"] or "CLOCK_ERROR"
                    )
                    self.assertEqual(snapshot["last_observed_ns"], before["last_observed_ns"])
                    self.assertEqual(snapshot["audit_attempts"], before["audit_attempts"])
                    self.assertEqual(snapshot["connection_attempts"], 1)
                    self.assertNotIn("private", str(snapshot))

    def test_later_clock_error_keeps_first_stop_reason_without_cleanup_allowance(self):
        for reason in ("WORK_DEADLINE", "PROTOCOL_ERROR", *REASONS):
            with self.subTest(reason=reason):
                window = self.make_window()
                observed = 100 + WORK_NS if reason == "WORK_DEADLINE" else 200
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns", return_value=observed
                ):
                    if reason == "WORK_DEADLINE":
                        window.work_remaining_ns()
                    elif reason == "PROTOCOL_ERROR":
                        self.assert_invalid(window.claim_connection)
                    else:
                        window.begin_cleanup(reason)
                before = window.snapshot()
                with patch(
                    "tokenizer_launch_window.time.monotonic_ns",
                    side_effect=RuntimeError("private"),
                ):
                    self.assert_invalid(window.cleanup_remaining_ns)
                with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
                    self.assertEqual(window.cleanup_remaining_ns(), 0)
                    self.assertEqual(window.work_remaining_ns(), 0)
                    self.assertIsNone(window.close())
                    clock.assert_not_called()
                snapshot = window.snapshot()
                self.assertEqual(snapshot["state"], "CLOSED")
                self.assertEqual(snapshot["stop_reason"], reason)
                self.assertEqual(snapshot["cleanup_deadline_ns"], before["cleanup_deadline_ns"])
                self.assertEqual(snapshot["last_observed_ns"], observed)
                self.assertIs(snapshot["clock_failed"], True)

    def call_from_worker(self, operation, *arguments):
        outcomes = []

        def invoke():
            try:
                operation(*arguments)
            except BaseException as error:
                outcomes.append(error)
            else:
                outcomes.append(None)

        worker = threading.Thread(target=invoke, name="MainThread")
        worker.start()
        worker.join()
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], self.module.LaunchWindowError)
        self.assertEqual(str(outcomes[0]), "tokenizer launch window invalid")

    def test_wrong_thread_construction_rejected_without_clock(self):
        with patch("tokenizer_launch_window.time.monotonic_ns") as clock:
            self.call_from_worker(self.module.LaunchWindow)
            clock.assert_not_called()

    def test_every_public_method_rejects_wrong_thread_before_clock_or_mutation(self):
        for state in ("NEW", "CONNECTING", "AUDITING", "CLEANUP", "CLOSED"):
            for method, arguments in (
                ("claim_connection", ()),
                ("claim_audit", ()),
                ("work_remaining_ns", ()),
                ("begin_cleanup", ("CANCELLED",)),
                ("begin_cleanup", (UntrustedReason(),)),
                ("cleanup_remaining_ns", ()),
                ("close", ()),
                ("snapshot", ()),
            ):
                with self.subTest(state=state, method=method):
                    window = self.make_window(state)
                    before = window.snapshot()
                    with patch(
                        "tokenizer_launch_window.time.monotonic_ns", return_value=101
                    ) as clock:
                        self.call_from_worker(getattr(window, method), *arguments)
                        clock.assert_not_called()
                    self.assertEqual(window.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
