import contextlib
import errno
import gc
import importlib
import importlib.util
import os
import select
import signal
import subprocess
import sys
import threading
import time
import unittest
import weakref
from unittest.mock import patch

from tokenizer_launch_window import LaunchWindow


FALSE_FLAGS = (
    "resource_limits_enforced", "root_write_limit_enforced", "deadline_enforced",
    "process_termination_enforced", "cleanup_verified", "benchmark_eligible",
    "formal_evaluation_ready", "net_benefit_established", "isolated",
    "all_possible_second_rounds_validated",
)


class OwnedChildTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tokenizer_owned_child"))
        self.module = importlib.import_module("tokenizer_owned_child")
        self.live_children = 0

    def assert_invalid(self, operation, *arguments):
        with self.assertRaises(self.module.OwnedChildError) as caught:
            operation(*arguments)
        self.assertEqual(str(caught.exception), "tokenizer owned child invalid")
        self.assertIsNone(caught.exception.__cause__)
        return caught.exception

    @contextlib.contextmanager
    def child(self, mode="live"):
        self.assertLess(self.live_children, 2)
        programs = {
            "live": (
                "import signal,time; signal.alarm(4); "
                "print('ready',flush=True); time.sleep(3)"
            ),
            "ignore": (
                "import signal,time; signal.alarm(4); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "print('ready',flush=True); time.sleep(3)"
            ),
            "exit": "import signal,sys; signal.alarm(4); print('ready',flush=True); sys.exit(7)",
            "controlled_exit": (
                "import select,signal,sys; signal.alarm(4); "
                "print('ready',flush=True); select.select([sys.stdin],[],[],3); sys.exit(7)"
            ),
        }
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", programs[mode]],
            env={}, close_fds=True, shell=False,
            stdin=subprocess.PIPE if mode == "controlled_exit" else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.live_children += 1
        try:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            self.assertTrue(ready, "fixture readiness deadline")
            self.assertEqual(os.read(process.stdout.fileno(), 6), b"ready\n")
            yield process
        finally:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
            finally:
                process.stdout.close()
                if process.stdin is not None:
                    process.stdin.close()
                self.live_children -= 1

    def await_reaped(self, owner):
        deadline = time.monotonic() + 2
        while not owner.poll():
            self.assertLess(time.monotonic(), deadline, "fixture reap deadline")
            time.sleep(0.005)

    @contextlib.contextmanager
    def synthetic(self, attach=True):
        process = object.__new__(subprocess.Popen)
        process._child_created = False
        process.pid = 123456789
        process.returncode = None
        with (
            patch("os.kill") as kill,
            patch("os.waitpid", return_value=(0, 0)) as waitpid,
            patch("signal.getsignal", return_value=signal.SIG_DFL),
        ):
            owner = self.module.OwnedChild()
            if attach:
                owner.attach(process)
                waitpid.reset_mock()
            yield owner, process, waitpid, kill

    def window(self, state="CLEANUP"):
        self.assertTrue(callable(getattr(self.module.OwnedChild, "terminate", None)))
        window = LaunchWindow()
        with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
            if state != "NEW":
                window.claim_connection()
            if state == "AUDITING":
                window.claim_audit()
            if state in ("CLEANUP", "CLOSED"):
                window.begin_cleanup("AUDIT_COMPLETE")
            if state == "CLOSED":
                window.close()
        return window

    def assert_snapshot_safe(self, owner):
        snapshot = owner.snapshot()
        self.assertEqual(len(snapshot), 19)
        self.assertEqual(snapshot["kind"], "LOCAL_OWNED_CHILD_LIFECYCLE_ONLY")
        self.assertEqual(snapshot["cleanup_status"], "CLEANUP_UNVERIFIED")
        for flag in FALSE_FLAGS:
            self.assertIs(snapshot[flag], False)
        for name, value in snapshot.items():
            self.assertIn(type(value), (str, bool, int, list, type(None)))
            if name == "signals_attempted":
                self.assertTrue(all(item in ("SIGTERM", "SIGKILL") for item in value))
        self.assertNotIn("123456789", str(snapshot))
        self.assertNotIn("private", str(snapshot))

    def test_construction_is_new_without_side_effects(self):
        with (
            patch("os.waitpid") as waitpid,
            patch("os.kill") as kill,
            patch("os.getpid") as getpid,
            patch("signal.getsignal") as getsignal,
            patch("time.monotonic_ns") as clock,
            patch("subprocess.Popen") as launch,
        ):
            owner = self.module.OwnedChild()
            self.assertEqual(owner.snapshot()["state"], "NEW")
            self.assertFalse(owner.snapshot()["child_reaped"])
            for boundary in (waitpid, kill, getpid, getsignal, clock, launch):
                boundary.assert_not_called()

    def test_attach_live_direct_child_and_poll_without_popen_methods(self):
        with self.child() as process:
            owner = self.module.OwnedChild()
            with (
                patch.object(process, "poll", side_effect=AssertionError("Popen.poll")),
                patch.object(process, "wait", side_effect=AssertionError("Popen.wait")),
                patch("os.waitpid", wraps=os.waitpid) as waitpid,
                patch("os.kill") as kill,
            ):
                self.assertIsNone(owner.attach(process))
                self.assertFalse(owner.poll())
                self.assertEqual(owner.snapshot()["state"], "OWNED")
                self.assertEqual(waitpid.call_args_list[0].args, (process.pid, os.WNOHANG))
                self.assertEqual(waitpid.call_count, 2)
                kill.assert_not_called()

    def test_natural_nonzero_exit_is_kernel_reaped_and_synchronizes_popen(self):
        with self.child("exit") as process:
            owner = self.module.OwnedChild()
            with patch("os.waitpid", wraps=os.waitpid) as waitpid:
                owner.attach(process)
                self.await_reaped(owner)
                self.assertGreaterEqual(waitpid.call_count, 1)
            self.assertEqual(process.returncode, 7)
            self.assertEqual(owner.snapshot()["exit_code"], 7)
            self.assertTrue(owner.snapshot()["child_reaped"])
            with patch("os.waitpid") as waitpid:
                self.assertTrue(owner.poll())
                self.assertTrue(owner.poll())
                waitpid.assert_not_called()

    def test_duplicate_handoff_and_repeated_attach_preserve_first_owner(self):
        with self.child() as process:
            first = self.module.OwnedChild()
            first.attach(process)
            before = first.snapshot()
            second = self.module.OwnedChild()
            self.assert_invalid(second.attach, process)
            self.assertEqual(second.snapshot()["state"], "LOST")
            self.assertEqual(second.snapshot()["failure"], "PROTOCOL_ERROR")
            with patch("os.waitpid") as waitpid:
                self.assert_invalid(first.attach, object())
                waitpid.assert_not_called()
            self.assertEqual(first.snapshot(), before)
            self.assertFalse(first.poll())

    def test_invalid_attach_consumes_only_attempt_and_new_lost_poll_rejects(self):
        for invalid in (None, 12, True, object()):
            with self.subTest(input_type=type(invalid).__name__):
                owner = self.module.OwnedChild()
                self.assert_invalid(owner.poll)
                self.assertEqual(owner.snapshot()["state"], "NEW")
                self.assert_invalid(owner.attach, invalid)
                snapshot = owner.snapshot()
                self.assertEqual(snapshot["state"], "LOST")
                self.assertTrue(snapshot["attach_attempted"])
                self.assertEqual(snapshot["failure"], "PROTOCOL_ERROR")
                with patch("os.waitpid") as waitpid:
                    self.assert_invalid(owner.attach, object())
                    self.assert_invalid(owner.poll)
                    waitpid.assert_not_called()
                self.assertEqual(owner.snapshot(), snapshot)

    def test_snapshot_exact_fresh_private_and_unverified(self):
        owner = self.module.OwnedChild()
        expected = {
            "kind": "LOCAL_OWNED_CHILD_LIFECYCLE_ONLY", "state": "NEW",
            "attach_attempted": False, "termination_attempted": False,
            "signals_attempted": [], "exit_code": None, "child_reaped": False,
            "failure": None, "cleanup_status": "CLEANUP_UNVERIFIED",
            **dict.fromkeys(FALSE_FLAGS, False),
        }
        self.assertEqual(owner.snapshot(), expected)
        snapshot = owner.snapshot()
        snapshot["signals_attempted"].append("private")
        snapshot.clear()
        self.assertEqual(owner.snapshot(), expected)

    def test_exact_popen_pid_and_returncode_validation(self):
        for pid in (True, False, 0, 1, -1, 2.0, None, "123", os.getpid()):
            with self.subTest(pid_type=type(pid).__name__, pid=pid):
                with self.synthetic(attach=False) as (owner, process, waitpid, kill):
                    process.pid = pid
                    self.assert_invalid(owner.attach, process)
                    self.assertEqual(owner.snapshot()["failure"], "PROTOCOL_ERROR")
                    waitpid.assert_not_called()
                    kill.assert_not_called()
                    process.pid = 123456789
                    self.assert_invalid(self.module.OwnedChild().attach, process)
        with self.synthetic(attach=False) as (owner, process, waitpid, kill):
            process.returncode = 0
            self.assert_invalid(owner.attach, process)
            waitpid.assert_not_called()
        class PopenSubclass(subprocess.Popen):
            pass
        process = object.__new__(PopenSubclass)
        process._child_created = False
        self.assert_invalid(self.module.OwnedChild().attach, process)

    def test_initial_nondefault_handler_rejects_before_wait(self):
        with self.synthetic(attach=False) as (owner, process, waitpid, kill):
            with patch("signal.getsignal", return_value=signal.SIG_IGN):
                self.assert_invalid(owner.attach, process)
            self.assertEqual(owner.snapshot()["failure"], "PROTOCOL_ERROR")
            waitpid.assert_not_called()
            kill.assert_not_called()

    def test_poll_identity_handler_and_returncode_drift_fail_closed(self):
        for drift in ("pid", "pid_bool", "returncode", "handler"):
            with self.subTest(drift=drift):
                with self.synthetic() as (owner, process, waitpid, kill):
                    if drift == "pid":
                        process.pid += 1
                    elif drift == "pid_bool":
                        process.pid = True
                    elif drift == "returncode":
                        process.returncode = 0
                    handler = signal.SIG_IGN if drift == "handler" else signal.SIG_DFL
                    with patch("signal.getsignal", return_value=handler):
                        self.assert_invalid(owner.poll)
                    self.assertEqual(owner.snapshot()["state"], "LOST")
                    self.assertEqual(owner.snapshot()["failure"], "OWNERSHIP_LOST")
                    self.assertFalse(owner.snapshot()["child_reaped"])
                    self.assert_invalid(owner.poll)
                    waitpid.assert_not_called()
                    kill.assert_not_called()

    def test_poll_wait_errors_and_cancellations_lose_ownership(self):
        faults = (
            (ChildProcessError("private"), "OWNERSHIP_LOST"),
            (OSError(errno.ECHILD, "private"), "OWNERSHIP_LOST"),
            (InterruptedError("private"), "WAIT_ERROR"),
            (OSError(errno.EIO, "private"), "WAIT_ERROR"),
            (RuntimeError("private"), "WAIT_ERROR"),
            (KeyboardInterrupt("private"), "CANCELLED"),
            (SystemExit("private"), "CANCELLED"),
            (GeneratorExit("private"), "CANCELLED"),
        )
        for attach in (True, False):
            for fault, reason in faults:
                with self.subTest(attach=attach, fault=type(fault).__name__):
                    with self.synthetic(attach=attach) as (owner, process, waitpid, kill):
                        waitpid.side_effect = fault
                        operation = owner.poll if attach else lambda: owner.attach(process)
                        if isinstance(fault, Exception):
                            caught = self.assert_invalid(operation)
                            self.assertTrue(caught.__suppress_context__)
                        else:
                            with self.assertRaises(BaseException) as caught:
                                operation()
                            self.assertIs(caught.exception, fault)
                        self.assertEqual(owner.snapshot()["state"], "LOST")
                        self.assertEqual(owner.snapshot()["failure"], reason)
                        self.assertFalse(owner.snapshot()["child_reaped"])
                        self.assert_invalid(owner.poll)
                        self.assertEqual(waitpid.call_count, 1)
                        kill.assert_not_called()
                        self.assert_snapshot_safe(owner)

    def test_poll_rejects_malformed_wait_results(self):
        for result in (
            None, [], [0, 0], (), (0,), (0, 0, 0), (False, 0), (0, False),
            (0, 1), (123456788, 0), (-1, 0), (123456789, -1),
            (123456789, 0x7F), (123456789, 0xFFFF), (123456789, 1.0),
        ):
            with self.subTest(result=result):
                with self.synthetic() as (owner, process, waitpid, kill):
                    waitpid.return_value = result
                    self.assert_invalid(owner.poll)
                    self.assertEqual(owner.snapshot()["failure"], "WAIT_ERROR")
                    self.assertEqual(owner.snapshot()["state"], "LOST")
                    self.assertIsNone(process.returncode)
                    kill.assert_not_called()

    def test_handoff_identity_is_weak_but_owner_reference_is_strong(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            reference = weakref.ref(process)
            del process
            gc.collect()
            self.assertIsNotNone(reference())
            waitpid.return_value = (123456789, 0)
            self.assertTrue(owner.poll())
            self.assert_invalid(self.module.OwnedChild().attach, reference())
        del owner
        gc.collect()
        self.assertIsNone(reference())

    def test_real_term_reaps_only_after_kernel_status(self):
        with self.child() as process:
            owner = self.module.OwnedChild()
            owner.attach(process)
            window = LaunchWindow()
            window.claim_connection()
            window.begin_cleanup("AUDIT_COMPLETE")
            self.assertTrue(callable(getattr(owner, "terminate", None)))
            with patch("os.kill", wraps=os.kill) as kill:
                self.assertTrue(owner.terminate(window))
            self.assertEqual(kill.call_args_list[0].args, (process.pid, signal.SIGTERM))
            self.assertEqual(kill.call_count, 1)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertEqual(owner.snapshot()["exit_code"], -signal.SIGTERM)
            self.assertTrue(owner.snapshot()["child_reaped"])
            self.assert_snapshot_safe(owner)

    def test_real_term_ignore_escalates_after_one_second_preserving_sibling(self):
        with self.child("ignore") as process, self.child() as sibling:
            owner = self.module.OwnedChild()
            owner.attach(process)
            window = LaunchWindow()
            window.claim_connection()
            window.begin_cleanup("AUDIT_COMPLETE")
            self.assertTrue(callable(getattr(owner, "terminate", None)))
            signals = []
            real_kill = os.kill

            def record_kill(pid, signum):
                self.assertEqual(pid, process.pid)
                signals.append((signum, time.monotonic_ns()))
                real_kill(pid, signum)

            with patch("os.kill", side_effect=record_kill):
                self.assertTrue(owner.terminate(window))
            self.assertEqual([entry[0] for entry in signals], [signal.SIGTERM, signal.SIGKILL])
            self.assertGreaterEqual(signals[1][1] - signals[0][1], 990_000_000)
            self.assertEqual(process.returncode, -signal.SIGKILL)
            self.assertIsNone(sibling.poll())
            self.assertEqual(owner.snapshot()["signals_attempted"], ["SIGTERM", "SIGKILL"])
            with patch("os.waitpid") as waitpid, patch("os.kill") as kill:
                self.assertTrue(owner.terminate(object()))
                self.assertTrue(owner.poll())
                waitpid.assert_not_called()
                kill.assert_not_called()

    def test_reaped_and_exited_termination_ignore_window_or_clock(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window()
            waitpid.return_value = (process.pid, 7 << 8)
            with patch("time.monotonic_ns") as clock:
                self.assertTrue(owner.terminate(window))
                self.assertTrue(owner.terminate(object()))
                self.assertEqual(waitpid.call_count, 1)
                clock.assert_not_called()
            kill.assert_not_called()
            self.assertTrue(owner.snapshot()["termination_attempted"])

    def test_invalid_windows_do_not_consume_or_observe(self):
        class WindowSubclass(LaunchWindow):
            pass
        for window in (None, object(), WindowSubclass(), self.window("NEW"),
                       self.window("CONNECTING"), self.window("AUDITING")):
            with self.synthetic() as (owner, process, waitpid, kill):
                before = owner.snapshot()
                with patch("time.monotonic_ns") as clock:
                    self.assert_invalid(owner.terminate, window)
                    clock.assert_not_called()
                self.assertEqual(owner.snapshot(), before)
                waitpid.assert_not_called()
                kill.assert_not_called()

    def test_closed_deadline_kills_once_and_later_poll_can_reap(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window("CLOSED")
            with patch("time.monotonic_ns") as clock, patch("time.sleep") as sleep:
                self.assertFalse(owner.terminate(window))
                self.assertEqual(owner.snapshot()["state"], "OWNED")
                self.assertEqual(owner.snapshot()["failure"], "DEADLINE")
                self.assertEqual(waitpid.call_count, 3)
                self.assert_invalid(owner.terminate, object())
                self.assertEqual(waitpid.call_count, 3)
                clock.assert_not_called()
                sleep.assert_not_called()
            kill.assert_called_once_with(process.pid, signal.SIGKILL)
            waitpid.return_value = (process.pid, int(signal.SIGKILL))
            self.assertTrue(owner.poll())
            self.assertEqual(owner.snapshot()["failure"], "DEADLINE")
            self.assert_snapshot_safe(owner)

    def test_exhausted_budget_kill_success_still_requires_wait_evidence(self):
        for reaped in (True, False):
            with self.subTest(reaped=reaped):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window()
                    before = window.snapshot()
                    if reaped:
                        waitpid.side_effect = [(0, 0), (0, 0), (process.pid, int(signal.SIGKILL))]
                    with (
                        patch("time.monotonic_ns", return_value=before["cleanup_deadline_ns"]),
                        patch("time.sleep") as sleep,
                    ):
                        self.assertIs(owner.terminate(window), reaped)
                        sleep.assert_not_called()
                    self.assertEqual(owner.snapshot()["child_reaped"], reaped)
                    self.assertEqual(window.snapshot()["cleanup_deadline_ns"],
                                     before["cleanup_deadline_ns"])
                    kill.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_synthetic_grace_and_sleep_caps_without_deadline_reset(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window()
            deadline = window.snapshot()["cleanup_deadline_ns"]
            now = [100]
            attempts = []
            sleeps = []

            def fake_sleep(seconds):
                self.assertGreater(seconds, 0)
                self.assertLessEqual(seconds, 0.01)
                self.assertLessEqual(seconds * 1_000_000_000, deadline - now[0])
                if len(attempts) == 1:
                    self.assertLessEqual(seconds * 1_000_000_000, 1_000_000_100 - now[0])
                sleeps.append(seconds)
                now[0] += round(seconds * 1_000_000_000)

            def fake_kill(pid, signum):
                attempts.append((pid, signum, now[0]))
                if signum == signal.SIGKILL:
                    waitpid.return_value = (pid, int(signum))

            kill.side_effect = fake_kill
            with patch("time.monotonic_ns", side_effect=lambda: now[0]), patch(
                "time.sleep", side_effect=fake_sleep
            ):
                self.assertTrue(owner.terminate(window))
            self.assertEqual(attempts, [
                (process.pid, signal.SIGTERM, 100),
                (process.pid, signal.SIGKILL, 1_000_000_100),
            ])
            self.assertEqual(len(sleeps), 100)
            self.assertEqual(window.snapshot()["cleanup_deadline_ns"], deadline)

    def test_short_cleanup_budget_clips_term_grace_and_last_sleep(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window()
            deadline = window.snapshot()["cleanup_deadline_ns"]
            now = [deadline - 15_000_000]
            sleeps = []

            def fake_sleep(seconds):
                sleeps.append(seconds)
                now[0] += round(seconds * 1_000_000_000)

            with patch("time.monotonic_ns", side_effect=lambda: now[0]), patch(
                "time.sleep", side_effect=fake_sleep
            ):
                self.assertFalse(owner.terminate(window))
            self.assertEqual(sleeps, [0.01, 0.005])
            self.assertEqual([entry.args for entry in kill.call_args_list], [
                (process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL),
            ])
            self.assertEqual(window.snapshot()["cleanup_deadline_ns"], deadline)
            self.assertEqual(owner.snapshot()["failure"], "DEADLINE")

    def test_fresh_poll_before_every_signal_prevents_lost_or_reaped_targets(self):
        for result, expected in (
            (ChildProcessError("private"), "OWNERSHIP_LOST"),
            ((123456789, 0), None),
        ):
            with self.subTest(result_type=type(result).__name__):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window()
                    waitpid.side_effect = [(0, 0), result]
                    with patch("time.monotonic_ns", return_value=100):
                        if expected:
                            self.assert_invalid(owner.terminate, window)
                        else:
                            self.assertTrue(owner.terminate(window))
                    self.assertEqual(owner.snapshot()["failure"], expected)
                    kill.assert_not_called()
                    self.assertEqual(waitpid.call_count, 2)

    def test_faults_emergency_kill_once_no_sleep_and_error_even_if_reaped(self):
        for boundary, reason in (
            ("clock", "CLOCK_ERROR"), ("term", "SIGNAL_ERROR"), ("sleep", "SLEEP_ERROR"),
        ):
            for reaped in (False, True):
                with self.subTest(boundary=boundary, reaped=reaped):
                    with self.synthetic() as (owner, process, waitpid, kill):
                        window = self.window()
                        clock_fault = RuntimeError("private")

                        def fake_kill(pid, signum):
                            if signum == signal.SIGTERM and boundary == "term":
                                raise ProcessLookupError("private")
                            if signum == signal.SIGKILL and reaped:
                                waitpid.return_value = (pid, int(signum))

                        kill.side_effect = fake_kill
                        with patch(
                            "time.monotonic_ns",
                            side_effect=clock_fault if boundary == "clock" else None,
                            return_value=100,
                        ), patch("time.sleep", side_effect=RuntimeError("private")) as sleep:
                            caught = self.assert_invalid(owner.terminate, window)
                            self.assertTrue(caught.__suppress_context__)
                            self.assertEqual(sleep.call_count, int(boundary == "sleep"))
                        self.assertEqual(owner.snapshot()["failure"], reason)
                        self.assertEqual(owner.snapshot()["child_reaped"], reaped)
                        self.assertEqual(
                            [entry.args[1] for entry in kill.call_args_list],
                            [signal.SIGKILL] if boundary == "clock"
                            else [signal.SIGTERM, signal.SIGKILL],
                        )
                        self.assert_snapshot_safe(owner)

    def test_closed_failed_clock_is_not_healthy_exhaustion(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window()
            with patch("time.monotonic_ns", side_effect=ValueError("private")):
                with self.assertRaises(ValueError):
                    window.cleanup_remaining_ns()
            with patch("time.monotonic_ns") as clock, patch("time.sleep") as sleep:
                self.assert_invalid(owner.terminate, window)
                clock.assert_not_called()
                sleep.assert_not_called()
            self.assertEqual(owner.snapshot()["failure"], "CLOCK_ERROR")
            kill.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_cancellation_at_clock_term_sleep_or_kill_preserves_original(self):
        for boundary in ("clock", "term", "sleep", "kill"):
            for exception_type in (KeyboardInterrupt, SystemExit, GeneratorExit):
                with self.subTest(boundary=boundary, exception=exception_type.__name__):
                    with self.synthetic() as (owner, process, waitpid, kill):
                        window = self.window("CLOSED" if boundary == "kill" else "CLEANUP")
                        original = exception_type("private original")

                        def fake_kill(pid, signum):
                            if (boundary == "term" and signum == signal.SIGTERM) or (
                                boundary == "kill" and signum == signal.SIGKILL
                            ):
                                raise original

                        kill.side_effect = fake_kill
                        with patch(
                            "time.monotonic_ns",
                            side_effect=original if boundary == "clock" else None,
                            return_value=100,
                        ), patch("time.sleep", side_effect=original) as sleep:
                            with self.assertRaises(BaseException) as caught:
                                owner.terminate(window)
                            self.assertIs(caught.exception, original)
                            self.assertEqual(sleep.call_count, int(boundary == "sleep"))
                        self.assertEqual(owner.snapshot()["failure"], "CANCELLED")
                        self.assertEqual(
                            [entry.args[1] for entry in kill.call_args_list].count(signal.SIGKILL), 1
                        )
                        before = waitpid.call_count
                        self.assert_invalid(owner.terminate, window)
                        self.assertEqual(waitpid.call_count, before)

    def test_emergency_preserves_earliest_cancellation_or_propagates_first_new_one(self):
        for original in (RuntimeError("private ordinary"), KeyboardInterrupt("private first")):
            for emergency_boundary in ("poll", "kill", "final_poll"):
                with self.subTest(original=type(original).__name__, emergency=emergency_boundary):
                    with self.synthetic() as (owner, process, waitpid, kill):
                        window = self.window()
                        later = SystemExit("private later")
                        if emergency_boundary == "poll":
                            waitpid.side_effect = [(0, 0), later]
                        elif emergency_boundary == "kill":
                            kill.side_effect = later
                        else:
                            waitpid.side_effect = [(0, 0), (0, 0), later]
                        with patch("time.monotonic_ns", side_effect=original), patch(
                            "time.sleep"
                        ) as sleep:
                            with self.assertRaises(BaseException) as caught:
                                owner.terminate(window)
                            self.assertIs(caught.exception,
                                          later if isinstance(original, Exception) else original)
                            sleep.assert_not_called()
                        self.assertEqual(owner.snapshot()["failure"],
                                         "CLOCK_ERROR" if isinstance(original, Exception) else "CANCELLED")

    def test_failed_kill_is_never_retried_even_during_emergency(self):
        for fault in (ProcessLookupError("private"), PermissionError("private")):
            with self.subTest(fault=type(fault).__name__):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window("CLOSED")
                    kill.side_effect = fault
                    self.assert_invalid(owner.terminate, window)
                    kill.assert_called_once_with(process.pid, signal.SIGKILL)
                    self.assertEqual(waitpid.call_count, 3)
                    self.assertEqual(owner.snapshot()["failure"], "SIGNAL_ERROR")
                    self.assertFalse(owner.snapshot()["child_reaped"])

    def test_observation_loss_in_termination_forbids_emergency_work(self):
        for fault in (InterruptedError("private"), ChildProcessError("private"),
                      KeyboardInterrupt("private")):
            with self.subTest(fault=type(fault).__name__):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window()
                    waitpid.side_effect = fault
                    with patch("time.monotonic_ns") as clock, patch("time.sleep") as sleep:
                        if isinstance(fault, Exception):
                            self.assert_invalid(owner.terminate, window)
                        else:
                            with self.assertRaises(BaseException) as caught:
                                owner.terminate(window)
                            self.assertIs(caught.exception, fault)
                        clock.assert_not_called()
                        sleep.assert_not_called()
                    self.assertEqual(waitpid.call_count, 1)
                    kill.assert_not_called()
                    self.assertEqual(owner.snapshot()["state"], "LOST")

    def test_wrong_thread_all_public_operations_are_inert(self):
        self.window()
        for state in ("NEW", "OWNED", "REAPED", "LOST"):
            with self.subTest(state=state):
                with self.synthetic(attach=state in ("OWNED", "REAPED")) as (
                    owner, process, waitpid, kill
                ):
                    if state == "REAPED":
                        waitpid.return_value = (process.pid, 0)
                        owner.poll()
                    elif state == "LOST":
                        self.assert_invalid(owner.attach, None)
                    before = owner.snapshot()
                    waitpid.reset_mock()
                    operations = (
                        self.module.OwnedChild, lambda: owner.attach(object()),
                        owner.poll, lambda: owner.terminate(object()), owner.snapshot,
                    )
                    for operation in operations:
                        outcomes = []

                        def invoke():
                            try:
                                operation()
                            except BaseException as error:
                                outcomes.append(error)

                        with patch("time.monotonic_ns") as clock:
                            worker = threading.Thread(target=invoke, name="MainThread")
                            worker.start()
                            worker.join(timeout=1)
                            self.assertFalse(worker.is_alive())
                            clock.assert_not_called()
                        self.assertEqual(len(outcomes), 1)
                        self.assertIsInstance(outcomes[0], self.module.OwnedChildError)
                        self.assertEqual(owner.snapshot(), before)
                    waitpid.assert_not_called()
                    kill.assert_not_called()

    def test_import_and_construction_never_touch_process_clock_or_payload_boundaries(self):
        specification = importlib.util.find_spec("tokenizer_owned_child")
        isolated = importlib.util.module_from_spec(specification)
        with contextlib.ExitStack() as stack:
            boundaries = [
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
                for name in (
                    "os.waitpid", "os.kill", "os.killpg", "os.getpid", "os.read",
                    "os.write", "os.close", "os.pipe", "os.open", "os.fork",
                    "time.monotonic_ns", "time.sleep", "signal.signal",
                    "signal.getsignal", "subprocess.Popen",
                )
            ]
            specification.loader.exec_module(isolated)
            owner = isolated.OwnedChild()
            self.assertEqual(owner.snapshot()["state"], "NEW")
            for boundary in boundaries:
                boundary.assert_not_called()

    def test_real_external_reap_is_ownership_loss_not_false_zero_exit(self):
        with self.child("controlled_exit") as process:
            owner = self.module.OwnedChild()
            owner.attach(process)
            process.stdin.write(b"\n")
            process.stdin.flush()
            deadline = time.monotonic() + 2
            while True:
                returned_pid, status = os.waitpid(process.pid, os.WNOHANG)
                if returned_pid == process.pid:
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            actual_exit = os.waitstatus_to_exitcode(status)
            self.assertEqual(actual_exit, 7)
            try:
                with patch("os.kill") as kill:
                    self.assert_invalid(owner.terminate, self.window("CLOSED"))
                    kill.assert_not_called()
                self.assertEqual(owner.snapshot()["failure"], "OWNERSHIP_LOST")
                self.assertEqual(owner.snapshot()["state"], "LOST")
                self.assertIsNone(owner.snapshot()["exit_code"])
                self.assertFalse(owner.snapshot()["child_reaped"])
            finally:
                process.returncode = actual_exit

    def test_new_and_lost_termination_do_not_touch_window(self):
        self.window()
        for lost in (False, True):
            with self.subTest(lost=lost):
                owner = self.module.OwnedChild()
                if lost:
                    self.assert_invalid(owner.attach, None)
                before = owner.snapshot()
                with (
                    patch.object(LaunchWindow, "snapshot") as snapshot,
                    patch("os.waitpid") as waitpid,
                    patch("os.kill") as kill,
                    patch("time.monotonic_ns") as clock,
                ):
                    self.assert_invalid(owner.terminate, object())
                    for boundary in (snapshot, waitpid, kill, clock):
                        boundary.assert_not_called()
                self.assertEqual(owner.snapshot(), before)

    def test_identity_drift_during_clock_check_prevents_term_and_kill(self):
        for before_signal in ("SIGTERM", "SIGKILL"):
            for drift in ("pid", "returncode", "handler"):
                with self.subTest(before_signal=before_signal, drift=drift):
                    with self.synthetic() as (owner, process, waitpid, kill):
                        window = self.window()
                        samples = []
                        handler = [signal.SIG_DFL]

                        def clock_sample():
                            samples.append(True)
                            if len(samples) == (1 if before_signal == "SIGTERM" else 2):
                                if drift == "pid":
                                    process.pid += 1
                                elif drift == "returncode":
                                    process.returncode = 0
                                else:
                                    handler[0] = signal.SIG_IGN
                            return 100 if len(samples) == 1 else 1_000_000_100

                        with patch("time.monotonic_ns", side_effect=clock_sample), patch(
                            "signal.getsignal", side_effect=lambda signum: handler[0]
                        ), patch("time.sleep") as sleep:
                            self.assert_invalid(owner.terminate, window)
                            sleep.assert_not_called()
                        self.assertEqual(owner.snapshot()["state"], "LOST")
                        self.assertEqual(owner.snapshot()["failure"], "OWNERSHIP_LOST")
                        self.assertEqual(kill.call_count, int(before_signal == "SIGKILL"))
                        self.assertNotIn("SIGKILL", owner.snapshot()["signals_attempted"])

    def test_after_kill_only_original_remaining_budget_is_used(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window()
            deadline = window.snapshot()["cleanup_deadline_ns"]
            with patch("time.monotonic_ns", side_effect=[
                100, 1_000_000_100, deadline - 5_000_000, deadline,
            ]) as clock, patch("time.sleep") as sleep:
                self.assertFalse(owner.terminate(window))
            self.assertEqual(clock.call_count, 4)
            sleep.assert_called_once_with(0.005)
            self.assertEqual(owner.snapshot()["signals_attempted"], ["SIGTERM", "SIGKILL"])
            self.assertEqual(kill.call_count, 2)
            self.assertEqual(waitpid.call_count, 7)
            self.assertEqual(window.snapshot()["cleanup_deadline_ns"], deadline)
            self.assertEqual(owner.snapshot()["failure"], "DEADLINE")

    def test_invalid_or_regressing_clock_after_term_triggers_no_sleep(self):
        for sample in (True, None, -1, 99, 100.0, "private"):
            with self.subTest(sample=sample):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window()
                    with patch("time.monotonic_ns", side_effect=[100, sample]), patch(
                        "time.sleep"
                    ) as sleep:
                        self.assert_invalid(owner.terminate, window)
                        sleep.assert_not_called()
                    self.assertEqual(owner.snapshot()["failure"], "CLOCK_ERROR")
                    self.assertEqual(window.snapshot()["last_observed_ns"], 100)
                    self.assertEqual(owner.snapshot()["signals_attempted"], ["SIGTERM", "SIGKILL"])

    def test_emergency_fault_does_not_replace_initial_ordinary_failure(self):
        for emergency_boundary in ("poll", "kill", "final_poll"):
            with self.subTest(emergency=emergency_boundary):
                with self.synthetic() as (owner, process, waitpid, kill):
                    window = self.window()
                    if emergency_boundary == "poll":
                        waitpid.side_effect = [(0, 0), ChildProcessError("private")]
                    elif emergency_boundary == "kill":
                        kill.side_effect = PermissionError("private")
                    else:
                        waitpid.side_effect = [(0, 0), (0, 0), InterruptedError("private")]
                    with patch("time.monotonic_ns", side_effect=RuntimeError("private")), patch(
                        "time.sleep"
                    ) as sleep:
                        self.assert_invalid(owner.terminate, window)
                        sleep.assert_not_called()
                    self.assertEqual(owner.snapshot()["failure"], "CLOCK_ERROR")
                    self.assertEqual(kill.call_count, int(emergency_boundary != "poll"))
                    self.assertEqual(waitpid.call_count, 2 if emergency_boundary == "poll" else 3)

    def test_signal_attempt_is_recorded_before_raising_and_snapshot_lists_are_fresh(self):
        with self.synthetic() as (owner, process, waitpid, kill):
            window = self.window("CLOSED")

            def failing_kill(pid, signum):
                self.assertEqual(owner.snapshot()["signals_attempted"], ["SIGKILL"])
                raise PermissionError("private")

            kill.side_effect = failing_kill
            self.assert_invalid(owner.terminate, window)
            snapshot = owner.snapshot()
            snapshot["signals_attempted"].clear()
            snapshot["failure"] = "private"
            self.assertEqual(owner.snapshot()["signals_attempted"], ["SIGKILL"])
            self.assertEqual(owner.snapshot()["failure"], "SIGNAL_ERROR")
            self.assert_snapshot_safe(owner)

    def test_window_prevalidation_cancellation_is_inert_and_allows_later_termination(self):
        for state in ("CLEANUP", "CLOSED"):
            for exception_type in (KeyboardInterrupt, SystemExit, GeneratorExit):
                with self.subTest(state=state, cancellation=exception_type.__name__):
                    with self.synthetic() as (owner, process, waitpid, kill):
                        window = self.window(state)
                        before = owner.snapshot()
                        private_before = dict(vars(owner))
                        process_before = dict(vars(process))
                        original = exception_type("private prevalidation cancellation")
                        with (
                            patch.object(window, "snapshot", side_effect=original) as snapshot,
                            patch("time.monotonic_ns") as clock,
                            patch("time.sleep") as sleep,
                            patch("os.getpid") as getpid,
                            patch("signal.getsignal") as getsignal,
                        ):
                            with self.assertRaises(BaseException) as caught:
                                owner.terminate(window)
                            self.assertIs(caught.exception, original)
                            snapshot.assert_called_once_with()
                            for boundary in (clock, sleep, getpid, getsignal, waitpid, kill):
                                boundary.assert_not_called()
                        self.assertEqual(owner.snapshot(), before)
                        self.assertEqual(vars(owner), private_before)
                        self.assertEqual(vars(process), process_before)

                        def collect_after_signal(pid, signum):
                            waitpid.return_value = (pid, int(signum))

                        kill.side_effect = collect_after_signal
                        with patch("time.monotonic_ns", return_value=100):
                            self.assertTrue(owner.terminate(window))
                        self.assertTrue(owner.snapshot()["termination_attempted"])
                        self.assertTrue(owner.snapshot()["child_reaped"])
                        self.assertIsNone(owner.snapshot()["failure"])
                        kill.assert_called_once_with(
                            process.pid, signal.SIGTERM if state == "CLEANUP" else signal.SIGKILL
                        )


if __name__ == "__main__":
    unittest.main()
