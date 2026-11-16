import importlib
import inspect
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from validation_contract import encoded
from validation_report import parse_outcome


ROOT = Path(__file__).parent


class TaskWorkerTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "validation_task_worker.py"
        self.assertTrue(self.path.is_file(), "task worker not implemented")

    def run_worker(self, source, args=None, **extra):
        request = {
            "request_id": "probe-0", "entry_point": "answer",
            "source": source, "args": [] if args is None else args, **extra,
        }
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(self.path)],
            input=encoded(request), capture_output=True, timeout=3,
        )

    def test_return_and_value_error(self):
        result = self.run_worker("def answer(values):\n    return sum(values)\n", [[2, 3]])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(parse_outcome(result.stdout), {"kind": "return", "value": 5})
        result = self.run_worker("def answer():\n    raise ValueError('private detail')\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(parse_outcome(result.stdout), {"kind": "raises", "type": "ValueError"})
        self.assertNotIn(b"private detail", result.stdout + result.stderr)

    def test_reject_private_request_fields(self):
        result = self.run_worker("def answer():\n    return 1\n", expected=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")

    def test_bad_source_and_wrong_exception_have_no_stack(self):
        for source in ("invalid syntax !", "def answer():\n    raise RuntimeError('secret')\n"):
            with self.subTest(source=source):
                result = self.run_worker(source)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, b"")

    def test_extra_output_is_not_discarded(self):
        result = self.run_worker("def answer():\n    print('fake pass')\n    return True\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"fake pass", result.stdout)
        with self.assertRaises(ValueError):
            parse_outcome(result.stdout)

    def test_duplicate_keys_rejected(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(self.path)],
            input=b'{"request_id":"a","request_id":"b"}',
            capture_output=True, timeout=3,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")


class RemoteCollectorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "validation_remote.py").is_file(),
                        "remote collector not implemented")
        self.module = importlib.import_module("validation_remote")

    def collect(self, source, input_raw=b"", **options):
        return self.module.collect_process(
            [sys.executable, "-I", "-S", "-B", "-c", source],
            input_raw, **options,
        )

    def test_echo_and_exact_schema(self):
        result = self.collect("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
                              b"hello")
        self.assertEqual(set(result), {
            "status", "returncode", "stdout_hex", "stderr_hex", "elapsed_ns",
            "process_reaped", "stdin_complete",
        })
        self.assertEqual(result["status"], "completed")
        self.assertEqual(bytes.fromhex(result["stdout_hex"]), b"hello")
        self.assertEqual(result["stderr_hex"], "")
        self.assertTrue(result["process_reaped"])
        self.assertTrue(result["stdin_complete"])

    def test_stderr_and_exit_are_not_normal_completion(self):
        result = self.collect("import sys; sys.stderr.write('diagnostic')")
        self.assertEqual(result["status"], "program_error")
        self.assertEqual(bytes.fromhex(result["stderr_hex"]), b"diagnostic")
        result = self.collect("raise SystemExit(7)")
        self.assertEqual(result["status"], "program_error")
        self.assertEqual(result["returncode"], 7)

    def test_ssh_diagnostics_are_preserved_separately_from_task_errors(self):
        self.assertIn("stderr_is_diagnostic",
                      inspect.signature(self.module.collect_process).parameters)
        result = self.collect("import sys; sys.stderr.write('SSH diagnostic')",
                              stderr_is_diagnostic=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(bytes.fromhex(result["stderr_hex"]), b"SSH diagnostic")

    def test_large_input_does_not_deadlock_or_drop_bytes(self):
        payload = b"payload" * 20000
        result = self.collect(
            "import sys; raw=sys.stdin.buffer.read(); print(len(raw))",
            payload, timeout_seconds=2,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(int(bytes.fromhex(result["stdout_hex"])), len(payload))

    def test_output_limit_is_bounded_and_process_reaped(self):
        for stream, field in (("stdout", "stdout_hex"), ("stderr", "stderr_hex")):
            with self.subTest(stream=stream):
                result = self.collect(
                    f"import sys; sys.{stream}.write('x'*1000000)",
                    stdout_limit=1024, stderr_limit=1024,
                )
                self.assertEqual(result["status"], "output_limit")
                self.assertLessEqual(len(bytes.fromhex(result[field])), 1025)
                self.assertTrue(result["process_reaped"])

    def test_timeout_and_expired_budget(self):
        started = time.monotonic()
        result = self.collect(
            "import time; print('READY', flush=True); time.sleep(5)",
            timeout_seconds=0.3,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(bytes.fromhex(result["stdout_hex"]), b"READY\n")
        self.assertTrue(result["process_reaped"])
        self.assertLess(time.monotonic() - started, 2)
        with self.assertRaises(ValueError):
            self.collect("pass", timeout_seconds=0)

    def test_launch_failure_and_early_exit(self):
        result = self.module.collect_process(["/nonexistent/tlf-program"], b"")
        self.assertEqual(result["status"], "launch_failed")
        self.assertIsNone(result["returncode"])
        result = self.collect("pass", b"x" * 1000000)
        self.assertNotEqual(result["status"], "completed")
        self.assertFalse(result["stdin_complete"])

    def test_bwrap_mounts_only_runtime_and_system_libraries(self):
        command = self.module.sandbox_argv("/runtime-origin", "worker", 9)
        self.assertEqual(command[0], "/usr/bin/bwrap")
        for flag in ("--unshare-all", "--die-with-parent", "--new-session", "--seccomp"):
            self.assertIn(flag, command)
        bindings = [command[index + 1:index + 3] for index, value in enumerate(command)
                    if value == "--ro-bind"]
        self.assertEqual(bindings, [
            ["/usr/lib", "/usr/lib"], ["/lib", "/lib"], ["/lib64", "/lib64"],
            ["/runtime-origin", "/runtime"],
        ])
        self.assertEqual(command[-6:], [
            "/runtime/bin/python3.12", "-I", "-S", "-B", "-c", "worker",
        ])

    def test_remote_response_validation_is_strict(self):
        self.assertTrue(hasattr(self.module, "response_is_valid"),
                        "remote response validation not implemented")
        for raw in (b"fake\n{}", b'{"kind":"return","value":0,"passed":true}',
                    b'{"kind":"return","value":NaN}', b'{"kind":"return","value":1.0}',
                    b'{"kind":"return","value":0,"value":1}',
                    b'{"kind":"raises","type":"RuntimeError"}'):
            with self.subTest(raw=raw):
                self.assertFalse(self.module.response_is_valid(raw))
        for raw in (b'{"kind":"return","value":true}',
                    b'{"kind":"return","value":{"key":[1,null,"ascii"]}}',
                    b'{"kind":"raises","type":"ValueError"}'):
            self.assertTrue(self.module.response_is_valid(raw))

    def test_clone3_compatibility_is_native_x86_64_only(self):
        self.assertTrue(hasattr(self.module, "syscall_number"),
                        "clone3 compatibility not implemented")
        self.assertEqual(self.module.syscall_number("clone3", -1, 0xC000003E), 435)
        self.assertEqual(self.module.syscall_number("clone3", 435, 0xC000003E), 435)
        for name, architecture in (("unknown", 0xC000003E), ("clone3", 0xC00000B7)):
            with self.subTest(name=name, architecture=architecture), self.assertRaises(RuntimeError):
                self.module.syscall_number(name, -1, architecture)

    def test_d01_requires_matching_preflight_runtime(self):
        self.assertTrue(hasattr(self.module, "verify_runtime"),
                        "pre-execution runtime binding not implemented")
        runtime = {"python_sha256": "verified"}
        self.module.verify_runtime("probe", runtime, None)
        self.module.verify_runtime("d01", runtime, runtime)
        for expected in (None, {"python_sha256": "changed"}):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.module.verify_runtime("d01", runtime, expected)


class RemoteBundleTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("validation_remote")
        self.bundle = {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE", "mode": "probe",
            "run_id": "a" * 32, "worker_source": "artificial worker",
            "requests": [{"request_id": "artificial-0", "source": "def answer(): return 1",
                          "entry_point": "answer", "args": []}],
        }

    def capture(self, raw, expected=None):
        output = io.StringIO()
        result = {
            "status": "completed", "returncode": 0,
            "stdout_hex": encoded({"kind": "return", "value": 1}).hex(),
            "stderr_hex": "", "elapsed_ns": 1, "process_reaped": True,
            "stdin_complete": True,
        }
        with tempfile.TemporaryFile() as filter_file, contextlib.ExitStack() as stack:
            descriptor = os.dup(filter_file.fileno())
            stack.enter_context(mock.patch.object(self.module.sys, "platform", "linux"))
            stack.enter_context(mock.patch.object(self.module, "no_new_privileges"))
            stack.enter_context(mock.patch.object(
                self.module, "seccomp_filter", return_value=(descriptor, "filter")))
            stack.enter_context(mock.patch.object(
                self.module.Path, "read_bytes", return_value=b"artificial binary"))
            stack.enter_context(mock.patch.object(
                self.module.os, "readlink", return_value="artificial namespace"))
            collector = stack.enter_context(mock.patch.object(
                self.module, "collect_process", return_value=result))
            stack.enter_context(contextlib.redirect_stdout(output))
            try:
                self.module.run_bundle(raw, expected_runtime=expected)
            finally:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                self.last_calls = collector.call_count
                self.last_output = output.getvalue()
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_ready_binds_formatted_raw_bytes(self):
        raw = (json.dumps(self.bundle, indent=2) + "\n").encode()
        events = self.capture(raw)
        self.assertEqual(events[0]["bundle_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.last_calls, 1)

    def test_matrix_counts_one_eight_and_thirty_two(self):
        runtime = self.capture(encoded(self.bundle))[0]["runtime"]
        self.bundle["mode"] = "matrix"
        for count in (1, 8, 32):
            self.bundle["requests"] = [self.bundle["requests"][0]] * count
            with self.subTest(count=count):
                try:
                    events = self.capture(encoded(self.bundle), runtime)
                except ValueError as error:
                    self.fail("matrix mode rejected: " + str(error))
                self.assertEqual(self.last_calls, count)
                self.assertEqual(events[-1]["received"], count)
                self.assertFalse(events[-1]["stopped"])
        self.bundle["requests"] *= 2
        with self.assertRaises(ValueError):
            self.capture(encoded(self.bundle), runtime)
        self.assertEqual(self.last_calls, 0)

    def test_matrix_runtime_gate_precedes_ready_and_collector(self):
        runtime = self.capture(encoded(self.bundle))[0]["runtime"]
        self.bundle["mode"] = "matrix"
        for expected in (None, {**runtime, "python_sha256": "changed"}):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                    ValueError, "runtime"):
                self.capture(encoded(self.bundle), expected)
            self.assertEqual(self.last_output, "")
            self.assertEqual(self.last_calls, 0)


if __name__ == "__main__":
    unittest.main()
