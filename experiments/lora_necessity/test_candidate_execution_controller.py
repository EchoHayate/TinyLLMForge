import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


READY = b"TLF_CANDIDATE_READY_V1\n"
PREFIX = """
import json
import os
import sys
import time
control = int(sys.argv[1])
os.write(control, b'TLF_CANDIDATE_READY_V1\\n')
payload = sys.stdin.buffer.read()
"""
RETURN = "os.write(control, b'{\"kind\":\"return\",\"value\":7}\\n')\n"


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("candidate_execution_controller.py").exists(),
                        "controller implementation missing")
        self.module = importlib.import_module("candidate_execution_controller")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.request = {
            "version": 1, "round": 1, "attempt_id": "a" * 32,
            "task_id": "D01", "arm": "base", "case_id": "P01",
            "entry_point": "identity", "source": "def identity(value):\n    return value\n",
            "args": [7],
        }

    def run_helper(self, source=PREFIX + RETURN, **options):
        return self.module.run_request(
            self.request, [sys.executable, "-I", "-S", "-B", "-c", source, "{control_fd}"],
            artifact_dir=self.root, cwd=self.root, env={"PATH": os.defpath},
            startup_timeout=options.pop("startup_timeout", 2.0),
            candidate_timeout=options.pop("candidate_timeout", 0.5),
            **options)

    def test_normal_result_and_parent_audit(self):
        result = self.run_helper()
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(result["receipt"]["outcome"], {"kind": "return", "value": 7})
        self.assertTrue(result["audit"]["input_complete"])
        self.assertTrue(result["audit"]["reaped"])
        self.assertTrue(result["audit"]["eof"])
        self.assertEqual(result["audit"]["returncode"], 0)
        self.assertGreater(result["audit"]["elapsed_seconds"], 0)
        self.assertEqual(json.loads((self.root / "receipt.json").read_bytes()), result["receipt"])

    def test_stdout_status_is_not_a_receipt(self):
        source = PREFIX + "print('{\"status\":\"timeout\",\"request_sha256\":\"fake\"}')\n" + RETURN
        self.assertEqual(self.run_helper(source)["receipt"]["status"], "completed")

    def test_value_error_outcome(self):
        source = PREFIX + "os.write(control, b'{\"kind\":\"raises\",\"type\":\"ValueError\"}\\n')"
        self.assertEqual(self.run_helper(source)["receipt"]["outcome"],
                         {"kind": "raises", "type": "ValueError"})

    def test_duplicate_result_is_program_error(self):
        self.assertEqual(self.run_helper(PREFIX + RETURN + RETURN)["receipt"]["status"],
                         "program_error")

    def test_nonzero_exit_cannot_report_success(self):
        self.assertEqual(self.run_helper(PREFIX + RETURN + "sys.exit(9)")["receipt"]["status"],
                         "program_error")

    def test_missing_result_is_program_error(self):
        self.assertEqual(self.run_helper(PREFIX)["receipt"]["status"], "program_error")

    def test_result_then_hang_is_timeout(self):
        result = self.run_helper(PREFIX + RETURN + "time.sleep(30)")
        self.assertEqual(result["receipt"]["status"], "timeout")
        self.assertTrue(result["audit"]["reaped"])
        self.assertLess(result["audit"]["elapsed_seconds"], 4)

    def test_stdout_flood_is_output_limit(self):
        source = PREFIX + "while True:\n    os.write(1, b'x' * 8192)\n"
        result = self.run_helper(source, output_limit=1024)
        self.assertEqual(result["receipt"]["status"], "output_limit")
        self.assertLessEqual(sum(path.stat().st_size for path in self.root.glob("*.bin")), 1025)

    def test_stderr_flood_is_output_limit(self):
        source = PREFIX + "while True:\n    os.write(2, b'x' * 8192)\n"
        self.assertEqual(self.run_helper(source, output_limit=1024)["receipt"]["status"],
                         "output_limit")

    def test_control_result_limit(self):
        source = PREFIX + "os.write(control, b'x' * 2000)\n"
        self.assertEqual(self.run_helper(source, result_limit=1024)["receipt"]["status"],
                         "output_limit")

    def test_bad_ready_is_infrastructure_failure(self):
        source = PREFIX.replace("TLF_CANDIDATE_READY_V1", "FAKE_CANDIDATE_READY_V1")
        with self.assertRaises(self.module.InfrastructureError):
            self.run_helper(source)
        self.assertFalse((self.root / "receipt.json").exists())
        self.assertFalse(json.loads((self.root / "observation.json").read_bytes())["ready"])

    def test_startup_timeout(self):
        with self.assertRaises(self.module.InfrastructureError):
            self.run_helper("import time; time.sleep(30)", startup_timeout=0.1)
        self.assertFalse((self.root / "receipt.json").exists())

    def test_source_is_not_sent_before_ready(self):
        source = """
import os
import select
import sys
control = int(sys.argv[1])
assert not select.select([sys.stdin], [], [], 0.1)[0]
os.write(control, b'TLF_CANDIDATE_READY_V1\\n')
payload = sys.stdin.buffer.read()
os.write(control, b'{"kind":"return","value":7}\\n')
"""
        self.assertEqual(self.run_helper(source)["receipt"]["status"], "completed")

    def test_large_input_is_written_without_blocking_reads(self):
        self.request["args"] = ["x" * 200000]
        self.assertEqual(self.run_helper()["receipt"]["status"], "completed")

    def test_input_backpressure_obeys_deadline(self):
        self.request["args"] = ["x" * 200000]
        source = PREFIX.replace("payload = sys.stdin.buffer.read()", "time.sleep(30)")
        with self.assertRaises(self.module.InfrastructureError):
            self.run_helper(source)
        audit = json.loads((self.root / "observation.json").read_bytes())
        self.assertFalse(audit["input_complete"])
        self.assertTrue(audit["reaped"])
        self.assertLess(audit["elapsed_seconds"], 4)

    def test_duplicate_claim_does_not_launch_again(self):
        self.run_helper()
        before = (self.root / "observation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_helper()
        self.assertEqual((self.root / "observation.json").read_bytes(), before)

    def test_invalid_configuration_does_not_consume_claim(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_helper(candidate_timeout=value)
        self.assertFalse((self.root / "claim.json").exists())

    def test_launch_failure_has_no_receipt(self):
        with self.assertRaises(self.module.InfrastructureError):
            self.module.run_request(
                self.request, ["/nonexistent/tlf-fixed-helper", "{control_fd}"],
                artifact_dir=self.root, cwd=self.root, env={})
        self.assertFalse((self.root / "receipt.json").exists())
        self.assertTrue((self.root / "observation.json").exists())

    def test_ready_to_send_delay_does_not_consume_candidate_window(self):
        base = self.module.selectors.DefaultSelector

        class DelayedSelector(base):
            def register(self, fileobj, events, data=None):
                if data == "input":
                    time.sleep(0.25)
                return super().register(fileobj, events, data)

        with mock.patch.object(self.module.selectors, "DefaultSelector", DelayedSelector):
            self.assertEqual(self.run_helper(candidate_timeout=0.15)["receipt"]["status"],
                             "completed")

    def test_failed_reap_does_not_get_second_full_attempt(self):
        original = self.module._terminate

        def failed_reap(process, timeout):
            original(process, timeout)
            raise self.module.InfrastructureError("simulated unconfirmed reap")

        with mock.patch.object(self.module, "_terminate", side_effect=failed_reap) as terminate:
            with self.assertRaises(self.module.InfrastructureError):
                self.run_helper(PREFIX + "time.sleep(30)", candidate_timeout=0.1)
        self.assertEqual(terminate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
